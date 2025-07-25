import os
import re
import time
import numpy as np
import torch
import dill
import threading
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R
import cv2
from diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer

from robot_env import RobotEnv, HUMAN, ROBOT, PRE_INTV, INTV, postprocess_action_mode
from hardware.my_device.macros import CAM_SERIAL
from util.episode_utils import EpisodeManager
from multi_robot.communication.socket_client import SocketClient
from multi_robot.utils.message_distillation import parse_message_regex

#About the keyboard listener: when robot is ctrled by agent,the keyboard listener in robot_node is used.When robot is ctrled by teleop,the keyboard listener in teleop_node is used.

class RobotNode:
    def __init__(self, config, rank: int, device_ids: List[int]):
        # self.teleop_minstep_over=True

        self.robot_id = config.robot_info.robot_id
        self.is_multi_robot_env = True if config.robot_info.num_robots > 1 else False
        self.config = config
        self.rank = rank
        self.device_ids = device_ids
        self.world_size = len(device_ids)
        self.device_id = device_ids[rank]
        self.device = f"cuda:{config.device_ids.split(',')[0]}"
        self.fps = 10
        self.robot_info_dict = self.config.robot_info.robot_info_dict

        # Initialize distributed training
        torch.distributed.init_process_group("nccl", rank=rank, world_size=self.world_size)
        
        # Initialize communication
        socket_ip = self.config.robot_info.socket_ip
        socket_port = self.config.robot_info.socket_port
        self.socket = SocketClient(socket_ip, socket_port, message_handler=self.handle_message)
        self.socket.start_connection()
        self.lock = threading.Lock()
        
        # Initialize components
        self._load_policy()
        self._setup_transformers()
        self._initialize_robot_env()
        self._initialize_episode_manager()
        self._initialize_replay_buffer()
        self.max_episode_length = self._calculate_max_episode_length()
        
        # Initialize failure detection module if specified
        self.failure_detection_module = None
        if hasattr(config, 'failure_detection_module') and config.failure_detection_module:
            self._initialize_failure_detection_module()
        
        # Set random seed
        self.seed = config.seed
        np.random.seed(self.seed)
        # print(f"DEBUG: seed: {self.seed}")
        
        # Setup output directory 
        self._setup_output_directory()
        # Extract round number for Sirius-specific logic
        self.num_round = self._extract_round_number()
        
        # State management
        self.robot_state = "idle"  # idle / teleop_controlled / agent_controlled / error
        self.teleop_id = 0
        self.last_query_robot = time.time() - 1
        self.inform_freq = 5  # Hz
        
        # Running state
        self.running = True
        self.episode_id = 0
        self.episode_idx = 0
        self.max_episode_length = self._calculate_max_episode_length()
        
        # Variables for sigma device control
        self.detach_tcp = None
        self.last_throttle = False
        self.last_p = None
        self.last_r = None
        self.diff_p_arr = None
        self.diff_r_arr = None
        self.width = None
        self.throttle = None
        self.delta_p_arr = None
        self.delta_r_arr = None
        self.rewind_key = False     #flag 1
        self.rewind_pos = None
        self.rewind_rot = None
        
        # Last predicted actions for rewinding
        self._last_predicted_abs_actions = None

        self.finish_episode = False
        
        # Scene alignment state
        self.scene_alignment_completed = False
        # self.robot_env.keyboard.kill_listener()
    
        
    def _load_policy(self):
        """Load policy model"""
        payload = torch.load(open(self.config.checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = payload['cfg']
        
        # Override some config values according to evaluation config
        self.cfg.policy.num_inference_steps = self.config.policy.num_inference_steps
        self.cfg.output_dir = self.config.output_dir
        if 'obs_encoder' in self.cfg.policy:
            self.cfg.policy.obs_encoder.pretrained_path = None
        
        # Initialize workspace
        import hydra
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg, self.rank, self.world_size, self.device_id, self.device)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # Get policy from workspace
        self.policy = workspace.model.module
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model.module
        
        self.policy.to(self.device)
        self.policy.eval()
        
        # Extract policy parameters
        self.To = self.policy.n_obs_steps
        self.Ta = self.policy.n_action_steps
        self.obs_feature_dim = self.policy.obs_feature_dim
        self.img_shape = self.cfg.task['shape_meta']['obs']['wrist_img']['shape']

        # Override Ta with evaluation config
        if hasattr(self.config, 'Ta'):
            self.Ta = self.config.Ta
    
    def _setup_transformers(self):
        """Setup rotation transformers"""
        self.action_dim = self.cfg.shape_meta['action']['shape'][0]
        self.action_rot_transformer = None
        self.obs_rot_transformer = None
        
        # Check if we need to transform rotation representation
        if 'rotation_rep' in self.cfg.shape_meta['action']:
            self.action_rot_transformer = RotationTransformer(
                from_rep='quaternion', 
                to_rep=self.cfg.shape_meta['action']['rotation_rep']
            )
        
        if 'ee_pose' in self.cfg.shape_meta['obs']:
            self.ee_pose_dim = self.cfg.shape_meta['obs']['ee_pose']['shape'][0]
            self.state_type = 'ee_pose'
            self.state_shape = self.cfg.task['shape_meta']['obs']['ee_pose']['shape']
            if 'rotation_rep' in self.cfg.shape_meta['obs']['ee_pose']:
                self.obs_rot_transformer = RotationTransformer(
                    from_rep='quaternion', 
                    to_rep=self.cfg.shape_meta['obs']['ee_pose']['rotation_rep']
                )
        else:
            self.ee_pose_dim = self.cfg.shape_meta['obs']['qpos']['shape'][0]
            self.state_type = 'qpos'
            self.state_shape = self.cfg.task['shape_meta']['obs']['qpos']['shape']
    
    def _initialize_robot_env(self):
        """Initialize robot environment"""
        self.robot_env = RobotEnv(
            camera_serial=CAM_SERIAL if not self.is_multi_robot_env else self.config.camera.serial, 
            img_shape=self.img_shape, 
            fps=self.fps,
            is_multi_robot_env=True,
            robot_id=self.robot_id,
            robot_info_dict=self.robot_info_dict
        )
    
    def _initialize_episode_manager(self):
        """Initialize episode manager"""
        num_samples = getattr(self.config, 'num_samples', 1)
        self.episode_manager = EpisodeManager(
            policy=self.policy,
            obs_rot_transformer=self.obs_rot_transformer,
            action_rot_transformer=self.action_rot_transformer,
            obs_feature_dim=self.obs_feature_dim,
            img_shape=self.img_shape,
            state_type=self.state_type,
            state_shape=self.state_shape,
            action_dim=self.action_dim,
            To=self.To,
            Ta=self.Ta,
            device=self.device,
            num_samples=num_samples
        )
    
    def _initialize_replay_buffer(self):
        """Initialize replay buffer"""
        zarr_path = os.path.join(self.config.train_dataset_path, 'replay_buffer.zarr')
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=None)
        
        # Add action_mode if not present
        if 'action_mode' not in self.replay_buffer.keys():
            self.replay_buffer.data['action_mode'] = np.full((self.replay_buffer.n_steps, ), HUMAN)
        
        # Add failure_indices if not present
        if 'failure_indices' not in self.replay_buffer.keys():
            self.replay_buffer.data['failure_indices'] = np.zeros((self.replay_buffer.n_steps, ), dtype=np.bool_)
    
    def _initialize_failure_detection_module(self):
        """Initialize failure detection module"""
        module_name = self.config.failure_detection_module
        
        if module_name:
            if module_name == 'action_inconsistency_ot':
                from failure_detection.action_inconsistency_ot import ActionInconsistencyOTModule
                self.failure_detection_module = ActionInconsistencyOTModule()
            elif module_name == 'baseline_logp':
                from failure_detection.baseline_logp import BaselineLogpModule
                self.failure_detection_module = BaselineLogpModule()
            else:
                raise ValueError(f"Unknown failure detection module: {module_name}")
            
            # Initialize the module
            self.failure_detection_module.initialize(
                cfg=self.config,
                device=self.device,
                policy=self.policy,
                replay_buffer=self.replay_buffer,
                episode_manager=self.episode_manager,
                Ta=self.Ta,
                To=self.To,
                ee_pose_dim=self.ee_pose_dim,
                img_shape=self.img_shape,
                obs_feature_dim=self.obs_feature_dim,
                max_episode_length=self.max_episode_length
            )
        else:
            self.failure_detection_module = None

    def _setup_output_directory(self):
        """Setup output directory"""
        self.save_img = False
        self.output_dir = os.path.join(self.config.output_dir, f"seed_{self.seed}")
        if os.path.isdir(self.output_dir):
            print(f"Output directory {self.output_dir} already exists, will not overwrite it.")
        else:
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
            self.save_img = True
        
        # Create save buffer directory
        os.makedirs(self.config.save_buffer_path, exist_ok=True)
    
    def _extract_round_number(self) -> int:
        """Extract round number from save buffer path"""
        match_round = re.search(r'round(\d)', self.config.save_buffer_path)
        if match_round:
            return int(match_round.group(1))
        return 0
    
    def _calculate_max_episode_length(self) -> int:
        """Calculate maximum episode length based on human demonstrations"""
        human_demo_indices = []
        for i in range(self.replay_buffer.n_episodes):
            episode_start = self.replay_buffer.episode_ends[i-1] if i > 0 else 0
            if np.any(self.replay_buffer.data['action_mode'][episode_start: self.replay_buffer.episode_ends[i]] == HUMAN):
                human_demo_indices.append(i)
        
        human_eps_len = []
        for i in human_demo_indices:
            human_episode = self.replay_buffer.get_episode(i)
            human_eps_len.append(human_episode['action'].shape[0])
        
        return int(max(human_eps_len) // self.Ta * self.Ta)

    def _run_rollout(self):
        """Main rollout loop"""
        try:
            while True: 
                # self.robot_env.keyboard.create_listener() if not self.robot_env.keyboard.listener else None

                # if self.robot_env.keyboard.quit:
                #     break

                print(f"Rollout episode: {self.episode_idx}")
                
                # Run single episode
                episode_data = self._run_single_episode()
                
                if episode_data is not None:
                    # Save episode to replay buffer
                    self.replay_buffer.add_episode(episode_data, compressors='disk')
                    self.episode_id = self.replay_buffer.n_episodes - 1
                    print(f'Saved episode {self.episode_id}')
                
                # Reset robot between episodes
                self.reset()   #2_reset
                # print("Reset!")
                
                self.episode_idx += 1
                
                # Check termination conditions
                if self._should_terminate():
                    break
                
                # For scene configuration reset
                time.sleep(5)
        
        finally:
            self._cleanup()

    def _run_single_episode(self):
        time.sleep(1)

        # Reset keyboard states
        with self.lock:
            self.finish_episode = False
        self.robot_state = "agent_controlled"  

        # Initialize episode buffers
        self.episode_buffers = {
            'tcp_pose': [],
            'joint_pos': [],
            'action': [],
            'action_mode': [],
            'wrist_cam': [],
            'side_cam': []
        }
         # Reset robot
        random_init_pose = None
        if getattr(self.config, 'random_init', False):
            random_init_pose = self.robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
        
        robot_state = self.reset(getattr(self.config, 'random_init', False), random_init_pose) #1_reset
         # Detach teleop device
        self.detach()

        # Reset observation history
        self.episode_manager.reset_observation_history()
        
        # Update initial observations
        for _ in range(self.To):
            self.episode_manager.update_observation(
                robot_state['policy_side_img'] / 255.0,
                robot_state['policy_wrist_img'] / 255.0,
                robot_state['tcp_pose'] if self.state_type == 'ee_pose' else robot_state['joint_pos']
            )
        
        # Initialize pose tracking
        if getattr(self.config, 'random_init', False) and random_init_pose is not None:
            self.episode_manager.initialize_pose(random_init_pose[:3], random_init_pose[3:])
        else:
            self.episode_manager.initialize_pose(self.robot_env.robot.init_pose[:3], self.robot_env.robot.init_pose[3:])
        
        # Scene alignment
        if self.save_img or not os.path.isfile(os.path.join(self.output_dir, f"side_{self.episode_idx}.png")):
            self.robot_env.save_scene_images(self.output_dir, self.episode_idx)
        else:
            self.request_scene_alignment(f"episode_start_{self.episode_idx}")
        
        # Initialize failure detection module step data
        if self.failure_detection_module:
            self.failure_detection_module.process_step({
                'step_type': 'episode_start',
                'episode_manager': self.episode_manager,
                'robot_state': robot_state
            })

        # Policy inference loop
        self.j = 0  # Episode timestep     

        while True:
            if self.j >= self.max_episode_length - self.Ta and self.robot_state == "agent_controlled":
                print("Maximum episode length reached, turning to human for help.")
                self.call_human_for_help("timeout")
                self.robot_state = "idle"

            # Handle robot state actions
            if self.robot_state == "agent_controlled":
                # self.robot_env.keyboard.create_listener() if not self.robot_env.keyboard.listener else None
                self.run_policy() 
                
            elif self.robot_state == "teleop_controlled":
                # print("==================sleep for teleop=======================")
                time.sleep(0.1)  # Prevent high CPU usage
            elif self.robot_state == "idle":
                # print("==================sleep for idle=======================")
                time.sleep(0.1)
                if self.rewind_key:
                    self.handle_prerewind_robot()
                    self.rewind_key = False

            with self.lock:
                finish_episode_flag = self.finish_episode
            
            if finish_episode_flag:
                print("===================break episode=======================")
                break

        # Finalize episode and return data
        if self.finish_episode:
            print("===================finalize episode 2=======================")
            episode_data = self._finalize_episode()
            return episode_data
        else:
            return None
            
    
    def run_policy(self):
        # self.robot_env.keyboard.create_listener() if not self.robot_env.keyboard.listener else None

        """Run policy inference loop"""
        while self.j < self.max_episode_length and self.robot_state == "agent_controlled" : # and not self.robot_env.keyboard.finish and not self.robot_env.keyboard.discard and not self.robot_env.keyboard.help:
            start_time = time.time()
            
            # Get robot state and observations
            robot_state = self.robot_env.get_robot_state()
            
            # Update observation history
            self.episode_manager.update_observation(
                robot_state['policy_side_img'] / 255.0,
                robot_state['policy_wrist_img'] / 255.0,
                robot_state['tcp_pose'] if self.state_type == 'ee_pose' else robot_state['joint_pos']
            )
            
            # Get action sequence for execution
            policy_obs = self.episode_manager.get_policy_observation()
            with torch.no_grad():
                if hasattr(self.policy, 'predict_action'):
                    if self.failure_detection_module and hasattr(self.failure_detection_module, 'needs_latent') and self.failure_detection_module.needs_latent:
                        curr_action, curr_latent = self.policy.predict_action(policy_obs, return_latent=True)
                    else:
                        curr_action = self.policy.predict_action(policy_obs)
                        curr_latent = None
                else:
                    curr_action = self.policy(policy_obs)
                    curr_latent = None
            
            # Get first Ta actions and execute on robot
            np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
            action_seq = np_action_dict['action']

            # Convert to absolute actions and store for rewinding
            predicted_abs_actions = np.zeros_like(action_seq[:, :, :8])
            
            # Execute action sequence
            for step in range(self.Ta):
                if step > 0:
                    start_time = time.time()
                
                # Get robot state
                if step == 0:
                    state_data = robot_state
                else:
                    state_data = self.robot_env.get_robot_state()
                
                # Get absolute action for this step
                deployed_action, gripper_action, curr_p, curr_r, curr_p_action, curr_r_action = \
                    self.episode_manager.get_absolute_action_for_step(action_seq, step)
                
                # Store predicted absolute actions for rewinding
                predicted_abs_actions[:, step] = np.concatenate((curr_p, curr_r.as_quat(scalar_first=True), gripper_action[:, np.newaxis]), -1)
                
                # Execute action on robot
                self.robot_env.deploy_action(deployed_action, gripper_action[0])
                
                # Save to episode buffers
                self.episode_buffers['wrist_cam'].append(state_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                self.episode_buffers['side_cam'].append(state_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                self.episode_buffers['tcp_pose'].append(state_data['tcp_pose'])
                self.episode_buffers['joint_pos'].append(state_data['joint_pos'])
                self.episode_buffers['action'].append(np.concatenate((curr_p_action, curr_r_action, [gripper_action[0]])))
                self.episode_buffers['action_mode'].append(ROBOT)
                
                # Update policy observation if needed
                if step >= self.Ta - self.To + 1:
                    self.episode_manager.update_observation(
                        state_data['policy_side_img'] / 255.0,
                        state_data['policy_wrist_img'] / 255.0,
                        state_data['tcp_pose'] if self.state_type == 'ee_pose' else state_data['joint_pos']
                    )
                
                time.sleep(max(1 / self.fps - (time.time() - start_time), 0))
                self.j += 1
            
            # Store predicted absolute actions for rewinding
            self._last_predicted_abs_actions = predicted_abs_actions
            
            # ================Detect failure===============
            if self.failure_detection_module:
                step_data = {
                    'step_type': 'policy_step',
                    'action_seq': action_seq,
                    'predicted_abs_actions': predicted_abs_actions,
                    'policy_obs': policy_obs,
                    'curr_latent': curr_latent,
                    'timestep': self.j,
                    'episode_manager': self.episode_manager
                }
                
                failure_step_data = self.failure_detection_module.process_step(step_data)  #对当前step，提交机器检查stepot_matching和action_inconsistency的申请，即对这两种情形都加入_async_queue
                
                failure_flag, self.failure_reason = self.failure_detection_module.detect_failure(  #阻塞式的，从FailureDetector.async_result_queue中获取所有结果，其中if result["task_type"] == "ot_matching"则调用failure_detector.submit_failure_detection_task，
                #其会对_async_queue来push"failure_detection"的请求;如果是"action_inconsistency"就会加入action_inconsistency_buffer;如果是"failure_detection",就会记录日志。
                    timestep=self.j,
                    max_episode_length=self.max_episode_length,
                    failure_step_data=failure_step_data
                )

                if failure_flag or self.j >= self.max_episode_length - self.Ta:
                    self.robot_state = "idle"         #temporarily set to idle for human judgement

                    if failure_flag:
                        print(f"Failure detected! Due to {self.failure_reason}")
                    else:
                        print("Maximum episode length reached")

                     # Send human check request
                    print("======================sending error to teleop======================== ") #print before keyboard is forbidden

                    if self.j >= self.max_episode_length - self.Ta:
                        self.call_human_for_help("timeout") #send message to teleop
                    else:
                        self.call_human_for_help("failure")
                    return  
            # ================Detect failure===============
            
            # Check for maximum episode length without failure detection
            elif self.j >= self.max_episode_length - self.Ta:
                self.robot_state = "idle"
                print("Maximum episode length reached")
                print("======================sending error to teleop 4=========================")
                self.call_human_for_help("timeout")
                return  # Return from run_policy to exit the while loop
        
        # This should only be reached if the while loop exits normally
        # print("DEBUG: run_policy() completed normally, while loop exited")
            
    def call_human_for_help(self,reason):
        # print(f"DEBUG: call_human_for_help started with reason: {reason}")
        # self.robot_env.keyboard.help = False  # prevent send message twice in the episode main loop 
        self.socket.send(f"NEED_HUMAN_CHECK_from_robot{self.robot_id}_for_{reason}") 
        # print("DEBUG: Message sent to socket")
        # self.robot_env.keyboard.kill_listener()
        # print("DEBUG: call_human_for_help completed")

    
    def get_message_patterns(self):
        """Get comprehensive message patterns for robust parsing"""
        # Pattern for floating point numbers (including negative and scientific notation)
        float_pattern = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
        
        patterns = {
            'READY': r'READY_for_state_check_by_human_with_teleop_id_\d+',
            'TELEOP_TAKEOVER_RESULT': r'TELEOP_TAKEOVER_RESULT_\w+_from_robot\d+',
            'CONTINUE_POLICY': r'CONTINUE_POLICY_\d+',
            'PLAYBACK_TRAJ': r'PLAYBACK_TRAJ_\d+',
            'TELEOP_CTRL_START': r'TELEOP_CTRL_START_\d+',
            'TELEOP_CTRL_STOP': r'TELEOP_CTRL_STOP_\d+_for_\w+',
            'THROTTLE_SHIFT': rf'THROTTLE_SHIFT_POSE_from_\d+_to_\d+:sigma:\[(?:{float_pattern}(?:,\s*{float_pattern})*)\],\[(?:{float_pattern}(?:,\s*{float_pattern})*)\]',
            'COMMAND': rf'COMMAND_from_\d+_to_\d+:sigma:\[(?:{float_pattern}(?:,\s*{float_pattern})*)\],\[(?:{float_pattern}(?:,\s*{float_pattern})*)\],{float_pattern},{float_pattern}',
            'REWIND_ROBOT': r'REWIND_ROBOT',
            'SCENE_ALIGNMENT_COMPLETED': r'SCENE_ALIGNMENT_COMPLETED'
        }
        return patterns
    
    def split_combined_messages(self, combined_msg):
        """Split combined messages using robust pattern matching"""
        if not combined_msg:
            return []
        
        # Single message case
        if not any(sep in combined_msg for sep in ['READY', 'TELEOP_TAKEOVER_RESULT', 'CONTINUE_POLICY', 
                                                   'PLAYBACK_TRAJ', 'TELEOP_CTRL_START', 'TELEOP_CTRL_STOP', 
                                                   'THROTTLE_SHIFT', 'COMMAND', 'REWIND_ROBOT', 'SCENE_ALIGNMENT_COMPLETED']):
            return [combined_msg]
        
        patterns = self.get_message_patterns()
        messages = []
        remaining = combined_msg
        
        while remaining:
            found_match = False
            best_match = None
            best_start = len(remaining)
            best_pattern_name = None
            
            # Find the earliest matching pattern
            for pattern_name, pattern in patterns.items():
                match = re.search(pattern, remaining)
                if match and match.start() < best_start:
                    best_match = match
                    best_start = match.start()
                    best_pattern_name = pattern_name
                    found_match = True
            
            if not found_match:
                # No more patterns found, add remaining as is
                if remaining.strip():
                    messages.append(remaining.strip())
                break
            
            # Add any content before the match
            if best_start > 0:
                prefix = remaining[:best_start].strip()
                if prefix:
                    messages.append(prefix)
            
            # Add the matched message
            messages.append(best_match.group(0))
            remaining = remaining[best_match.end():]
        
        return [msg for msg in messages if msg.strip()]
    
    def handle_message(self, raw_msg):
        # flag = 0
        # if "THROTTLE_SHIFT_POSE" in raw_msg:
        #     flag = 1
        # print("=========================raw_msg:{}=======================".format(raw_msg))
        """Handle received messages"""
        if "COMMAND" not in raw_msg:
            pass
            # print(f"DEBUG: Raw message received: {repr(raw_msg)}")
        message_list = self.split_combined_messages(raw_msg)
        if "COMMAND" not in raw_msg:
            pass
            # print(f"DEBUG: Split into {len(message_list)} messages: {[repr(msg) for msg in message_list]}")
        
        for message in message_list:
            if message.startswith("READY"):
                self.get_ready_takeover(message)

            elif message.startswith("TELEOP_TAKEOVER_RESULT"): #direct result without human teleoperation,succ/fail
                self.detach()
                # print(f"DEBUG: Setting finish_episode=True in thread {threading.current_thread().name}")
                if "FAILURE" in message:
                    self.handle_failure()
                with self.lock:
                    self.finish_episode = True  #finish_episode has greater priority over robot_state
                print("===================finish episode_1=======================")
                # print(f"DEBUG: After setting, finish_episode={self.finish_episode}")

            elif message.startswith("CONTINUE_POLICY"):   #direct result without human teleoperation
                self.handle_ctn()
            elif message.startswith("PLAYBACK_TRAJ"):
                self.playback_traj()
            elif message.startswith("TELEOP_CTRL_START"):  #lead to resume and transform
                self.start_being_teleoped()
            elif message.startswith("COMMAND"):
                # print(f"DEBUG: Processing COMMAND message: {repr(message)}")
                self.process_teleop_command(message)
            elif message.startswith("TELEOP_CTRL_STOP"):  #direct result after human teleoperation,cancel/accept/continue
                self.stop_teleop(message)
            elif message.startswith("THROTTLE_SHIFT"):
                self.process_throttle_info(message)
            elif message.startswith("REWIND_ROBOT"):
                self.rewind_key = True
                
            elif message.startswith("SCENE_ALIGNMENT_COMPLETED"):
                self.handle_scene_alignment_completed(message)
            else:
                print(f"未知命令: {message}")
            # if flag == 1:
            #     raise RuntimeError("=====================throttle not detacted====================================")
    
    def handle_ctn(self):
        self.robot_state = "agent_controlled"
        if self.j >= self.max_episode_length - self.Ta: # after hum
            print("Maximum episode length reached, turning to human for help.")
            self.call_human_for_help("timeout")
            self.robot_state = "idle"

        print("False Positive failure! Continue policy rollout.")
        if self.failure_reason == 'action inconsistency' and self.failure_detection_module.enable_action_inconsistency:
            self.failure_detection_module.failure_detector.expert_action_threshold = np.inf
            print("Reset the action inconsistency threshold to infinity temporarily")
    
    def handle_failure(self):
        if hasattr(self.failure_detection_module, 'failure_logs'):
            if self.failure_detection_module.failure_logs:  # Check if dictionary is not empty
                self.failure_detection_module.failure_logs.popitem()
        elif hasattr(self.failure_detection_module, 'failure_indices'):
            if self.failure_detection_module.failure_indices:  # Check if list is not empty
                self.failure_detection_module.failure_indices.pop()
                    
    
    def get_ready_takeover(self, message):
        """Prepare for takeover"""
        templ = "READY_for_state_check_by_human_with_teleop_id_{}"
        teleop_id = parse_message_regex(message, templ)[0]
        self.teleop_id = teleop_id

    def process_throttle_info(self, msg):
        """Process throttle shift information"""
        print(f"========================={msg}=============================")
        pattern = r"THROTTLE_SHIFT_POSE_from_(\d+)_to_(\d+):sigma:\[([^\]]+)\],\[([^\]]+)\]"
        match = re.match(pattern, msg)
        
        if not match:
            raise ValueError(f"Invalid command format: {msg}")
            
        teleop_id = match.group(1)
        rbt_id = match.group(2)
        diff_p_str = match.group(3)
        diff_r_str = match.group(4)
        
        # Parse arrays
        self.delta_p_arr = np.array([float(x.strip()) for x in diff_p_str.split(",")])
        self.delta_r_arr = np.array([float(x.strip()) for x in diff_r_str.split(",")])
    
    def reset_teleop_cmd(self):
        """Reset teleop command variables"""
        self.last_p = None
        self.last_r = None
        self.diff_p_arr = None
        self.diff_r_arr = None
        self.width = None
        self.throttle = None
        self.delta_p_arr = None
        self.delta_r_arr = None
    
    def handle_prerewind_robot(self):
        # Perform rewinding if needed
        print("Starting rewind process...")
        if self.failure_detection_module and hasattr(self, 'episode_buffers') and hasattr(self, 'j'):
            try:
                self.j, self.rewind_pos, self.rewind_rot = self._rewind_robot(self.episode_buffers, self.j)  #TODO: remember to restore rewind
                print(f"Rewind completed. New timestep: {self.j}")
            except Exception as e:
                print(f"Error during rewind: {e}")
                print("Continuing without rewind...")
        else:
            print("No failure detection module or missing episode data, skipping rewind")
        
        # Send rewind completion message
        print("Rewind process completed")
        rewind_complete_msg = f"REWIND_COMPLETED_{self.robot_id}"
        self.socket.send(rewind_complete_msg)
    
    def request_scene_alignment(self, context_info):
        """Request scene alignment from teleop node"""
        print(f"Requesting scene alignment for: {context_info}")
        
        # Extract episode index from context if it's episode_start
        if context_info.startswith("episode_start_"):
            episode_idx = context_info.split("_")[-1]
            # Send request to teleop node to handle user interaction
            align_msg = f"SCENE_ALIGNMENT_REQUEST_{self.robot_id}_{context_info}"
            self.socket.send(align_msg)
            # Start local image display in robot end
            self.start_scene_alignment_display(episode_idx)
        else:
            print(f"Unknown context for scene alignment: {context_info}")
        
        print("Scene alignment completed")
    
    def handle_scene_alignment_completed(self, message):
        """Handle scene alignment completion notification"""
        self.scene_alignment_completed = True
    
    def request_scene_alignment_with_reference(self, ref_side_cam, ref_wrist_cam): #rewind
        """Request scene alignment with reference images from teleop node"""
        print("Requesting scene alignment with reference images from teleop")
        
        # Send simple scene alignment request to teleop (no image data)
        align_msg = f"SCENE_ALIGNMENT_WITH_REF_REQUEST_{self.robot_id}_rewind"
        self.socket.send(align_msg)
        
        # Start local image display with reference images in robot end
        ref_side_img = cv2.cvtColor(ref_side_cam, cv2.COLOR_RGB2BGR)
        ref_wrist_img = cv2.cvtColor(ref_wrist_cam, cv2.COLOR_RGB2BGR)
        self.start_scene_alignment_display_with_reference(ref_side_img, ref_wrist_img)  #这个函数内容是：一直显示照片，要等收到SCENE_ALIGNMENT_COMPLETED才会停止显示照片。
        
        print("Scene alignment with reference completed")
    
    def start_scene_alignment_display(self, episode_idx):
        """Start scene alignment display in robot end, wait for teleop confirmation"""
        # Load reference images from saved files
        ref_side_img = cv2.imread(f"{self.output_dir}/side_{episode_idx}.png")
        ref_wrist_img = cv2.imread(f"{self.output_dir}/wrist_{episode_idx}.png")

        # Use the restored local scene alignment display
        if ref_side_img is not None and ref_wrist_img is not None:
            self.start_scene_alignment_display_with_reference(ref_side_img, ref_wrist_img, raw=True)
        else:
            print(f"⚠️  Could not load reference images for episode {episode_idx}")
            print("Proceeding without scene alignment...")
            time.sleep(2.0)  # Brief pause to allow manual alignment
    
    def start_scene_alignment_display_with_reference(self, ref_side_img, ref_wrist_img, raw=False):
        """Start scene alignment display with provided reference images"""
        print("Scene alignment display with reference started - waiting for teleop confirmation")  
        cv2.namedWindow("Wrist", cv2.WINDOW_AUTOSIZE)

        print("=================start image display=================")
        
        # Reset completion flag
        self.scene_alignment_completed = False
        
        # Add timeout mechanism to prevent infinite waiting
        start_time = time.time()
        timeout_duration = 60.0
        
        # Display images until teleop confirms completion or timeout
        while not self.scene_alignment_completed:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check for timeout
            if elapsed_time > timeout_duration:
                print(f"⚠️  TIMEOUT after {timeout_duration} seconds. Auto-continuing...")
                print("🤖 Network may be disconnected, proceeding automatically")
                break
            
            try:
                state_data = self.robot_env.get_robot_state()
                if raw:
                    side_img = state_data['side_img_raw']
                    wrist_img = state_data['wrist_img_raw']
                else:
                    side_img = cv2.cvtColor(state_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                    wrist_img = cv2.cvtColor(state_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                # Blend current and reference images
                blended_side = (np.array(side_img) * 0.5 + np.array(ref_side_img) * 0.5).astype(np.uint8)
                blended_wrist = (np.array(wrist_img) * 0.5 + np.array(ref_wrist_img) * 0.5).astype(np.uint8)
                
                cv2.imshow("Side", blended_side)
                cv2.imshow("Wrist", blended_wrist)
                cv2.waitKey(1)  # Non-blocking, just for OpenCV refresh
                
                time.sleep(0.1)  # Prevent high CPU usage
                
            except Exception as e:
                print(f"⚠️  Error during scene alignment display: {e}")
                print("Continuing despite error...")
                time.sleep(0.5)
        
        cv2.destroyAllWindows()
        print("Scene alignment display with reference completed")
          
    
    def start_being_teleoped(self):
        """Start being teleoperated"""

        print("Start being teleoperated, state switched to 'TELEOP_CONTROLLED'")
        # Get current pose for human teleop
        self.last_p = self.rewind_pos if self.rewind_pos is not None else self.episode_manager.last_p[0]
        self.last_r = self.rewind_rot if self.rewind_rot is not None else self.episode_manager.last_r[0]
        
        # Transform sigma device from current robot pose
        translate = self.last_p - self.detach_tcp[0:3]
        rotation = (R.from_quat(self.detach_tcp[3:], scalar_first=True).inv() * self.last_r).as_quat(scalar_first=True)
        
        # Resume sigma device
        self.send_resume_sigma() #1_resume
        self.send_transform_sigma(translate, rotation)
        # time.sleep(0.5)
        self.robot_state = "teleop_controlled"
    
    def stop_teleop(self, message):
        """Stop teleoperation"""
        self.robot_state = "idle"
        print("Teleoperation terminated, state switched to 'AGENT_CONTROLLED'.")
        templ = "TELEOP_CTRL_STOP_{}_for_{}"
        rbt_id, stop_event = parse_message_regex(message, templ)
        
        # Reset pose for policy after human intervention
        self.episode_manager.initialize_pose(self.last_p, self.last_r.as_quat(scalar_first=True))

        self.reset_teleop_cmd()
        self.detach()
        assert stop_event in ["cancel","accept","continue"]

        if stop_event != "continue":  #success or failure
            if stop_event=="cancel":
                self.handle_failure()
            with self.lock:
                self.finish_episode = True   #finish_episode has greater priority over robot_state
            print("===================finish episode_2=======================")
        else:
            self.robot_state = "agent_controlled"
            self.handle_ctn()
    
    def process_teleop_command(self, message):
        """Process teleoperation command"""
        state_data = self.robot_env.get_robot_state()
        tcp_pose = state_data['tcp_pose']
        joint_pos = state_data['joint_pos']

        if "sigma" in message:
            # print(f"DEBUG: Attempting to parse sigma command: {repr(message)}")
            pattern = r"COMMAND_from_(\d+)_to_(\d+):sigma:\[([^\]]+)\],\[([^\]]+)\],([^,]+),([^,]+)"
            match = re.match(pattern, message)
            
            if not match:
                # print(f"DEBUG: Pattern match failed for message: {repr(message)}")
                # print(f"DEBUG: Expected pattern: COMMAND_from_X_to_Y:sigma:[array1],[array2],value1,value2")
                # print(f"DEBUG: Message length: {len(message)}")
                
                # Try to identify where the message might be truncated
                if message.count('[') != message.count(']'):
                    pass
                    # print("DEBUG: Message appears to have unmatched brackets - likely truncated")
                if not message.strip().endswith(']') and ',' in message[-20:]:
                    pass
                    # print("DEBUG: Message appears to end mid-value - likely truncated")
                
                raise ValueError(f"Invalid command format: {message[:100]}{'...' if len(message) > 100 else ''}")
                
            # print(f"DEBUG: Successfully parsed command groups: {match.groups()}")
            teleop_id = match.group(1)
            rbt_id = match.group(2)
            diff_p_str = match.group(3)
            diff_r_str = match.group(4)
            width = match.group(5)
            throttle = match.group(6)
            
            try:
                # Parse arrays with better error handling
                self.diff_p_arr = np.array([float(x.strip()) for x in diff_p_str.split(",")])
                self.diff_r_arr = np.array([float(x.strip()) for x in diff_r_str.split(",")])
                self.width = float(width)
                self.throttle = float(throttle)
                print("=========================throttle:{}=======================".format(self.throttle))
                # print(f"DEBUG: Parsed arrays - diff_p: {self.diff_p_arr.shape}, diff_r: {self.diff_r_arr.shape}")
            except ValueError as e:
                # print(f"DEBUG: Failed to parse numeric values: {e}")
                # print(f"DEBUG: diff_p_str: {repr(diff_p_str)}")
                # print(f"DEBUG: diff_r_str: {repr(diff_r_str)}")
                # print(f"DEBUG: width: {repr(width)}")
                # print(f"DEBUG: throttle: {repr(throttle)}")
                raise ValueError(f"Failed to parse numeric values from command: {e}")
            
            abs_p = self.robot_env.robot.init_pose[:3] + self.diff_p_arr
            abs_r = R.from_quat(self.robot_env.robot.init_pose[3:7], scalar_first=True).inv() * R.from_quat(self.diff_r_arr, scalar_first=True)
            curr_p_action = abs_p - self.last_p
            curr_r_action = self.last_r.inv() * abs_r
            self.last_p = abs_p
            self.last_r = abs_r
            
            # Handle throttle detach
            if self.throttle < -0.9:
                print(f"============================1111111Last throttle: {self.last_throttle}=============================")
                if not self.last_throttle:
                    self.detach()
                    self.last_throttle = True
                return

            if self.last_throttle:
                print(f"============================22222222Last throttle: {self.last_throttle}=============================")
                self.last_throttle = False
                self.send_resume_sigma(during_teleop=True)  #2_resume
                while self.delta_p_arr is None or self.delta_r_arr is None: #delta_p_arr is not none when receive a msg called THROTTLE_SHIFT_POSE,which is triggered by teleop's handle_sigma_resume
                    time.sleep(0.001)
                return
                
            # Execute command on robot
            if self.robot_state == "teleop_controlled": 
                # Deploy action to robot
                self.robot_env.deploy_action(
                    np.concatenate((abs_p, abs_r.as_quat(scalar_first=True))),
                    self.width
                )
                self.robot_env.gripper.move_from_sigma(self.width)
                gripper_action = self.robot_env.gripper.max_width * self.width / 1000
                
                # Save demo data for return
                teleop_data = {
                    'policy_wrist_img': state_data['policy_wrist_img'],
                    'policy_side_img': state_data['policy_side_img'],
                    'demo_wrist_img': state_data['demo_wrist_img'],
                    'demo_side_img': state_data['demo_side_img'],
                    'tcp_pose': tcp_pose,
                    'joint_pos': joint_pos,
                    'action': np.concatenate((curr_p_action, curr_r_action.as_quat(scalar_first=True), [gripper_action])),
                    'action_mode': INTV
                }
    #==============simulate robot_env.human_teleop_step==============
                # Update observation history with latest state
                self.episode_manager.update_observation(
                    teleop_data['policy_side_img'] / 255.0,
                    teleop_data['policy_wrist_img'] / 255.0,
                    teleop_data['tcp_pose'] if self.state_type == 'ee_pose' else teleop_data['joint_pos']
                )
                
                # Store demo data
                self.episode_buffers['wrist_cam'].append(teleop_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                self.episode_buffers['side_cam'].append(teleop_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                self.episode_buffers['tcp_pose'].append(teleop_data['tcp_pose'])
                self.episode_buffers['joint_pos'].append(teleop_data['joint_pos'])
                self.episode_buffers['action'].append(teleop_data['action'])
                self.episode_buffers['action_mode'].append(teleop_data['action_mode'])
                self.j += 1
        # self.teleop_minstep_over = True
        
                        
    def inform_robot_state(self):
        """Periodically report robot state"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_query_robot >= 1.0 / self.inform_freq:
                self.last_query_robot = current_time
                self.socket.send(f"INFORM_ROBOT_STATE_{self.robot_id}_{self.robot_state}")
            time.sleep(0.01)
    
    def reset(self,random_init=False, random_init_pose=None):
        state=self.robot_env.reset_robot(random_init, random_init_pose)
        self.send_reset_sigma()
        tcp = state["tcp_pose"]
        translate = tcp[0:3] - self.robot_env.home_pose[0:3]
        rotation = (R.from_quat(self.robot_env.home_pose[3:], scalar_first=True).inv() * R.from_quat(tcp[3:],scalar_first=True)).as_quat(scalar_first=True)
        self.send_transform_sigma(translate,rotation)
        return state
        
    def detach(self):
        self.detach_tcp = self.robot_env.robot.get_tcp_pose()
        print(f"Detaching sigma at TCP position: {self.detach_tcp}")
        self.send_detach_sigma()
    
    def send_transform_sigma(self, translate, rotation):
        """Send transform message to sigma device"""
        msg = f"SIGMA_TRANSFORM_from_{self.robot_id}_{translate.tolist() + rotation.tolist()}_to_{self.teleop_id}".encode()
        self.socket.send(msg)
    
    def send_detach_sigma(self):
        """Send detach message to sigma device"""
        msg = f"SIGMA_of_{self.teleop_id}_DETACH_from_{self.robot_id}".encode()
        self.socket.send(msg)
    
    def send_reset_sigma(self):
        """Send reset message to sigma device"""
        msg = f"SIGMA_of_{self.teleop_id}_RESET_from_{self.robot_id}".encode()
        self.socket.send(msg)
    
    def send_resume_sigma(self, during_teleop=False):
        """Send resume message to sigma device"""
        if during_teleop:
            msg = f"SIGMA_of_{self.teleop_id}_RESUME_from_{self.robot_id}_DURING_TELEOP".encode()
        else:
            msg = f"SIGMA_of_{self.teleop_id}_RESUME_from_{self.robot_id}".encode()
        self.socket.send(msg)
    
    def _rewind_robot(self, episode_buffers: Dict[str, List], j) -> Tuple[int, np.ndarray, R]:
        """Rewind the robot for human intervention"""
        print("Rewinding robot...")
        
        # Get the last predicted absolute actions for rewinding
        if not hasattr(self, '_last_predicted_abs_actions') or self._last_predicted_abs_actions is None:
            print("No previous actions to rewind from, using current episode manager pose")
            return j, self.episode_manager.last_p[0], self.episode_manager.last_r[0]
        
        curr_timestep = j
        
        # Use the last target action for rewinding instead of the current robot state
        curr_tcp = self._last_predicted_abs_actions[0, self.Ta-1, :7]
        curr_pos = curr_tcp[:3]
        curr_rot = R.from_quat(curr_tcp[3:], scalar_first=True)
        
        # Let failure detection module determine rewinding behavior
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'should_stop_rewinding'):
            rewind_steps, prev_side_cam, prev_wrist_cam, curr_pos, curr_rot = self._rewind_with_failure_detection(episode_buffers, curr_timestep, curr_pos, curr_rot)
        else:
            rewind_steps, prev_side_cam, prev_wrist_cam, curr_pos, curr_rot = self._rewind_simple(episode_buffers, curr_timestep, curr_pos, curr_rot)
        
        j -= rewind_steps
        
        # Prepare for human intervention by showing reference scene
        if rewind_steps > 0:
            print("Please reset the scene and press 'c' in teleop node to go on to human intervention")
            self.request_scene_alignment_with_reference(prev_side_cam, prev_wrist_cam)
        
        return j, curr_pos, curr_rot
    
    def _rewind_with_failure_detection(self, episode_buffers: Dict[str, List], curr_timestep: int, curr_pos: np.ndarray, curr_rot: R) -> int:
        """Rewind with failure detection module guidance"""
        rewind_steps = 0
        j = curr_timestep
        if hasattr(self.failure_detection_module, 'greedy_ot_cost'):
            total_greedy_ot_cost = torch.sum(self.failure_detection_module.greedy_ot_cost[:curr_timestep // self.Ta])
        
        for i in range(curr_timestep):
            # Check if failure detection module says to stop rewinding
            if j % self.Ta == 0 and j > 0:
                if self.failure_detection_module.should_stop_rewinding(j, episode_buffers, total_greedy_ot_cost):
                    print("Failure detection module says to stop rewinding.")
                    break
                
                # Rewind the OT plan if it's an action inconsistency + OT module
                if hasattr(self.failure_detection_module, '_rewind_ot_plan'):
                    self.failure_detection_module._rewind_ot_plan(j)
            
            # Rewind one step
            curr_pos, curr_rot, prev_side_cam, prev_wrist_cam = self._rewind_single_step(episode_buffers, curr_pos, curr_rot)
            j -= 1
            rewind_steps += 1
        
        return rewind_steps, prev_side_cam, prev_wrist_cam, curr_pos, curr_rot
    
    def _rewind_simple(self, episode_buffers: Dict[str, List], curr_timestep: int, curr_pos: np.ndarray, curr_rot: R) -> int:
        """Simple rewinding with step limit"""
        rewind_steps = 0
        j = curr_timestep
        
        for i in range(curr_timestep):
            # Simple stop condition: limit to 3 Ta-step chunks
            if j % self.Ta == 0 and j > 0:
                if i // self.Ta >= 3:
                    print("Stop rewinding (reached 3 Ta-step limit).")
                    break
            
            # Rewind one step
            curr_pos, curr_rot, prev_side_cam, prev_wrist_cam = self._rewind_single_step(episode_buffers, curr_pos, curr_rot)
            j -= 1
            rewind_steps += 1
        
        return rewind_steps, prev_side_cam, prev_wrist_cam, curr_pos, curr_rot
    
    def _rewind_single_step(self, episode_buffers: Dict[str, List], curr_pos: np.ndarray, curr_rot: R) -> Tuple[np.ndarray, R]:
        """Rewind a single step"""
        # Get previous action data to rewind
        prev_wrist_cam = episode_buffers['wrist_cam'].pop()
        prev_side_cam = episode_buffers['side_cam'].pop()
        episode_buffers['tcp_pose'].pop()
        episode_buffers['joint_pos'].pop()
        prev_action = episode_buffers['action'].pop()
        episode_buffers['action_mode'].pop()
        
        # Let failure detection module handle additional buffer cleanup
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'rewind_step_cleanup'):
            self.failure_detection_module.rewind_step_cleanup()
        
        # Rewind robot by applying inverse action
        curr_pos, curr_rot = self.robot_env.rewind_robot(curr_pos, curr_rot, prev_action)
        
        return curr_pos, curr_rot, prev_side_cam, prev_wrist_cam
    
    def _finalize_episode(self) -> Dict[str, Any]:
        """Finalize episode and return episode data"""
        episode = dict()
        episode['wrist_cam'] = np.stack(self.episode_buffers['wrist_cam'], axis=0)
        episode['side_cam'] = np.stack(self.episode_buffers['side_cam'], axis=0)
        episode['tcp_pose'] = np.stack(self.episode_buffers['tcp_pose'], axis=0)
        episode['joint_pos'] = np.stack(self.episode_buffers['joint_pos'], axis=0)
        episode['action'] = np.stack(self.episode_buffers['action'], axis=0)
        
        # Process action mode
        if getattr(self.config, 'post_process_action_mode', False):
            episode['action_mode'] = postprocess_action_mode(np.array(self.episode_buffers['action_mode']))
        else:
            episode['action_mode'] = np.array(self.episode_buffers['action_mode'])
        
        assert episode['action_mode'].shape[0] % self.Ta == 0, "A Ta-step chunking is required for the entire demo"
        
        # Finalize failure detection
        if self.failure_detection_module:
            failure_episode_data = self.failure_detection_module.finalize_episode({
                'episode': episode,
                'episode_id': self.episode_id
            })
            episode.update(failure_episode_data)
        else:
            # Default: no failure indices
            episode['failure_indices'] = np.zeros((episode['action_mode'].shape[0],), dtype=np.bool_)
        
        return episode

    def _should_terminate(self) -> bool:
        """Check if rollout should terminate"""
        # Sirius-specific termination condition
        if hasattr(self.config, 'sirius_termination') and self.config.sirius_termination:
            human_actions = np.sum(self.replay_buffer.data['action_mode'] == HUMAN)
            intervention_actions = np.sum(self.replay_buffer.data['action_mode'] == INTV)
            
            if intervention_actions * 3 / self.num_round >= human_actions:
                return True
            
            progress = intervention_actions * 300 / self.num_round / human_actions if human_actions > 0 else 0
            print(f"Current progress: {progress:.2f}%")
        
        return False
    
    def _cleanup(self):
        """Cleanup resources"""
        # Save the replay buffer
        save_zarr_path = os.path.join(self.config.save_buffer_path, 'replay_buffer.zarr')
        self.replay_buffer.save_to_path(save_zarr_path)
        
        # Cleanup failure detection module
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'cleanup'):
            self.failure_detection_module.cleanup()
        
        print("Saved replay buffer to", save_zarr_path)
        torch.distributed.destroy_process_group() 
    
    def _main_thread(self):
         # Start state report thread
        self.inform_thread = threading.Thread(target=self.inform_robot_state, daemon=True)
        self.inform_thread.start()

        self._run_rollout()
    