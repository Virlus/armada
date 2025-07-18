import os
import re
import time
import numpy as np
import torch
import dill
import threading
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R

from diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer

from robot_env import RobotEnv, HUMAN, ROBOT, PRE_INTV, INTV
from hardware.my_device.macros import CAM_SERIAL
from util.episode_utils import EpisodeManager
from multi_robot.communication.socket_client import SocketClient
from multi_robot.utils.message_distillation import parse_message_regex

class RobotNode:
    def __init__(self, config, rank: int, device_ids: List[int], robot_id, socket_ip, socket_port):
        self.robot_id = robot_id
        self.config = config
        self.rank = rank
        self.device_ids = device_ids
        self.world_size = len(device_ids)
        self.device_id = device_ids[rank]
        self.device = f"cuda:{config.device_ids.split(',')[0]}"
        self.fps = 10
        
        # Initialize communication
        self.socket = SocketClient(socket_ip, socket_port, message_handler=self.handle_message)
        self.socket.start_connection()
        self.lock = threading.Lock()
        
        # Initialize components
        self._load_policy()
        self._setup_transformers()
        self._initialize_robot_env()
        self._initialize_episode_manager()
        self._initialize_replay_buffer()
        self._initialize_failure_detection()
        
        # Set random seed
        self.seed = config.seed
        np.random.seed(self.seed)
        
        # Setup output directory 
        self._setup_output_directory()
        # Extract round number for Sirius-specific logic
        self.num_round = self._extract_round_number()
        
        # State management
        self.robot_state = "idle"  # idle / teleop_controlled / agent_controlled / error
        self.teleop_id = None
        self.last_query_robot = time.time() - 1
        self.inform_freq = 5  # Hz
        
        # Running state
        self.running = True
        self.episode_idx = 0
        self.max_episode_length = self._calculate_max_episode_length()
        
        # Demo buffer for trajectory recording and rewinding
        self.demo_buffer = []  # List[Dict[str, List]]
        
        # Variables for sigma device control
        self.detach_tcp = None
        self.last_throttle = False
        self.diff_p_arr = None
        self.diff_r_arr = None
        self.width = None
        self.throttle = None
        self.delta_p_arr = None
        self.delta_r_arr = None
        
        # Last predicted actions for rewinding
        self._last_predicted_abs_actions = None
        
        # Start state report thread
        self.inform_thread = threading.Thread(target=self.inform_robot_state, daemon=True)
        self.inform_thread.start()
        
    
        
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
            camera_serial=CAM_SERIAL, 
            img_shape=self.img_shape, 
            fps=self.fps
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
    
    def _initialize_failure_detection(self):
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
    
    def get_separator_pattern(self):
        """Get message separator pattern"""
        separators = [
            "READY",
            "TELEOP_TAKEOVER_RESULT",
            "CONTINUE_POLICY",
            "PLAYBACK_TRAJ",
            "TELEOP_CTRL_START",
            "COMMAND",
            "TELEOP_CTRL_STOP",
            "THROTTLE_SHIFT"
        ]
        sorted_seps = sorted(separators, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_seps))
        return re.compile(f"({pattern})")
    
    def split_combined_messages(self, combined_msg):
        """Split combined messages"""
        if not combined_msg:
            return []

        pattern = self.get_separator_pattern()
        parts = []
        matches = list(pattern.finditer(combined_msg))
        if not matches:
            return [combined_msg]

        if matches[0].start() > 0:
            parts.append(combined_msg[:matches[0].start()])

        for i, match in enumerate(matches):
            start = match.start()
            next_start = matches[i + 1].start() if i < len(matches) - 1 else len(combined_msg)
            content = combined_msg[start:next_start]
            parts.append(content)
        return parts
    
    def handle_message(self, raw_msg):
        """Handle received messages"""
        message_list = self.split_combined_messages(raw_msg)
        for message in message_list:
            if message.startswith("READY"):
                self.get_ready_takeover(message)
            elif message.startswith("TELEOP_TAKEOVER_RESULT"):
                self.robot_state = "idle"
                self.detach()
                self.back_to_agent_ctrl(message, continue_current=False)
            elif message.startswith("CONTINUE_POLICY"):
                self.robot_state = "idle"
                self.detach()
                self.back_to_agent_ctrl(message, continue_current=True)
            elif message.startswith("PLAYBACK_TRAJ"):
                self.playback_traj()
            elif message.startswith("TELEOP_CTRL_START"):
                self.start_being_teleoped()
            elif message.startswith("COMMAND"):
                self.process_teleop_command(message)
            elif message.startswith("TELEOP_CTRL_STOP"):
                self.stop_teleop(message)
            elif message.startswith("THROTTLE_SHIFT"):
                self.process_throttle_info(message)
            else:
                print(f"Unknown command: {message}")
    
    def get_ready_takeover(self, message):
        """Prepare for takeover"""
        templ = "READY_for_state_check_by_human_with_teleop_id_{}"
        teleop_id = parse_message_regex(message, templ)[0]
        self.teleop_id = teleop_id
        self.detach()
    
    def back_to_agent_ctrl(self, message, continue_current):
        """Return to agent control"""
        print("Human take-over terminated, state switched to 'AGENT_CONTROLLED'.")
        if not continue_current:
            print("============================Next episode!!========================================")
            
            # Reset robot
            self.robot_env.reset_robot()
            self.robot_state = "agent_controlled"
            
            # Reset sigma device
            self.send_reset_sigma()
            
            # Transform sigma device based on current pose
            tcp = self.robot_env.robot.get_robot_state()[0]
            translate = tcp[0:3] - self.robot_env.home_pose[0:3]
            rotation = (R.from_quat(self.robot_env.home_pose[3:], scalar_first=True).inv() * 
                        R.from_quat(tcp[3:], scalar_first=True)).as_quat(scalar_first=True)
            self.send_transform_sigma(translate, rotation)
        
        self.detach()
        self.robot_state = "agent_controlled"
        self.run_policy()
    
    def start_being_teleoped(self):
        """Start being teleoperated"""
        print("Start being teleoperated, state switched to 'TELEOP_CONTROLLED'")
        
        # Transform sigma device from current robot pose to match
        tcp = self.robot_env.robot.get_robot_state()[0]
        translate = tcp[0:3] - self.detach_tcp[0:3]
        rotation = (R.from_quat(self.detach_tcp[3:], scalar_first=True).inv() * 
                    R.from_quat(tcp[3:], scalar_first=True)).as_quat(scalar_first=True)
        
        # Resume sigma device
        self.send_resume_sigma()
        time.sleep(0.1)
        self.send_transform_sigma(translate, rotation)
        self.robot_state = "teleop_controlled"
    
    def stop_teleop(self, message):
        """Stop teleoperation"""
        print("Teleoperation terminated, state switched to 'AGENT_CONTROLLED'.")
        templ = "TELEOP_CTRL_STOP_{}_for_{}"
        rbt_id, stop_event = parse_message_regex(message, templ)
        
        self.reset_teleop_cmd()
        if stop_event == "continue":
            self.robot_state = "idle"
            self.detach()
            self.back_to_agent_ctrl(message, continue_current=True)
        elif stop_event == "accept":
            self.robot_state = "idle"
            self.detach()
            self.back_to_agent_ctrl("SUCCESS", continue_current=False)
        else:
            assert stop_event == "cancel"
            self.robot_state = "idle"
            self.detach()
            self.back_to_agent_ctrl("FAILURE", continue_current=False)
    
    def process_throttle_info(self, msg):
        """Process throttle shift information"""
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
        self.diff_p_arr = None
        self.diff_r_arr = None
        self.width = None
        self.throttle = None
        self.delta_p_arr = None
        self.delta_r_arr = None
    
    def process_teleop_command(self, message):
        """Process teleoperation command"""
        if "sigma" in message:
            pattern = r"COMMAND_from_(\d+)_to_(\d+):sigma:\[([^\]]+)\],\[([^\]]+)\],([^,]+),([^,]+)"
            match = re.match(pattern, message)
            
            if not match:
                raise ValueError(f"Invalid command format: {message}")
                
            teleop_id = match.group(1)
            rbt_id = match.group(2)
            diff_p_str = match.group(3)
            diff_r_str = match.group(4)
            width = match.group(5)
            throttle = match.group(6)
            
            # Parse arrays
            self.diff_p_arr = np.array([float(x.strip()) for x in diff_p_str.split(",")])
            self.diff_r_arr = np.array([float(x.strip()) for x in diff_r_str.split(",")])
            self.width = float(width)
            self.throttle = float(throttle)
            
            # Handle throttle detach
            if self.throttle < -0.9:
                if not self.last_throttle:
                    self.detach()
                    self.last_throttle = True
                return
                
            if self.last_throttle:
                while self.delta_p_arr is None or self.delta_r_arr is None:
                    time.sleep(0.001)
                
                self.last_throttle = False
                self.send_resume_sigma(during_teleop=True)
                return
                
            # Execute command on robot
            if self.robot_state == "teleop_controlled":
                # Get absolute positions from relative commands
                abs_p = self.robot_env.robot.init_pose[:3] + self.diff_p_arr
                abs_r = (R.from_quat(self.robot_env.robot.init_pose[3:7], scalar_first=True).inv() * 
                        R.from_quat(self.diff_r_arr, scalar_first=True))
                
                # Deploy action to robot
                self.robot_env.deploy_action(
                    np.concatenate((abs_p, abs_r.as_quat(scalar_first=True))),
                    self.width
                )
    
    def playback_traj(self):
        """Playback trajectory"""
        print("Playback trajectory")
        if self.demo_buffer:
            if len(self.demo_buffer) > 0:
                if len(self.demo_buffer[-1]["action"]) > 0:
                    # Get current robot pose
                    curr_pose = self.robot_env.robot.get_tcp_pose()
                    curr_pos = curr_pose[:3]
                    curr_rot = R.from_quat(curr_pose[3:], scalar_first=True)
                    
                    # Rewind robot with last action
                    self.robot_env.rewind_robot(curr_pos, curr_rot, self.demo_buffer[-1]["action"][-1])
                    
                    # Remove the action from demo buffer
                    for k, v in self.demo_buffer[-1].items():
                        del self.demo_buffer[-1][k][-1]
                        
    def inform_robot_state(self):
        """Periodically report robot state"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_query_robot >= 1.0 / self.inform_freq:
                self.last_query_robot = current_time
                self.socket.send(f"INFORM_ROBOT_STATE_{self.robot_id}_{self.robot_state}")
            time.sleep(0.01)

    def run_policy(self):
        """Run policy inference loop"""

        episode_buffers = {
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
        
        robot_state = self.robot_env.reset_robot(getattr(self.config, 'random_init', False), random_init_pose)

        # Reset observation history
        self.episode_manager.reset_observation_history()
        
        # Initialize observation history with EpisodeManager
        for idx in range(self.To):
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
            self.robot_env.align_scene_with_file(self.output_dir, self.episode_idx)

        
        # Initialize failure detection module step data
        if self.failure_detection_module:
            self.failure_detection_module.process_step({
                'step_type': 'episode_start',
                'episode_manager': self.episode_manager,
                'robot_state': robot_state
            })
        # Detach teleop device
        self.detach()

        # Policy inference
        j = 0  # Episode timestep         #TODO:last check at here
        
        while j < self.max_episode_length and self.robot_state == "agent_controlled": #TODO:here correspond to _run_policy_inference_loop in real_robot_runner.py
            start_time = time.time()
            
            # Get robot state
            robot_state = self.robot_env.get_robot_state()
            
            # Update observation history
            self.episode_manager.update_observation(
                robot_state['policy_side_img'] / 255.0,
                robot_state['policy_wrist_img'] / 255.0,
                robot_state['tcp_pose'] if self.state_type == 'ee_pose' else robot_state['joint_pos']
            )
            
            # Get policy observation and predict action
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
            
            # Process action
            np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
            action_seq = np_action_dict['action']
            
            # Store demo buffers
            episode_buffers['wrist_cam'].append(robot_state['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            episode_buffers['side_cam'].append(robot_state['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            episode_buffers['tcp_pose'].append(robot_state['tcp_pose'])
            episode_buffers['joint_pos'].append(robot_state['joint_pos'])
            
            # Convert to absolute actions and store for rewinding
            predicted_abs_actions = np.zeros_like(action_seq[:, :, :8])
            
            # Process all Ta steps
            for step in range(self.Ta):
                if step > 0:
                    start_time = time.time()
                
                # Get state data
                if step == 0:
                    state_data = robot_state
                else:
                    state_data = self.robot_env.get_robot_state()
                
                # Get absolute action for this step
                deployed_action, gripper_action, curr_p, curr_r, curr_p_action, curr_r_action = \
                    self.episode_manager.get_absolute_action_for_step(action_seq, step)
                
                # Store predicted absolute actions for rewinding
                predicted_abs_actions[:, step] = np.concatenate(
                    (curr_p, curr_r.as_quat(scalar_first=True), gripper_action[:, np.newaxis]), -1
                )
                
                # Execute action on robot
                self.robot_env.deploy_action(deployed_action, gripper_action[0])
                
                # Add to episode buffers
                episode_buffers['action'].append(np.concatenate((curr_p_action, curr_r_action, [gripper_action[0]])))
                episode_buffers['action_mode'].append(ROBOT)
                
                # Update observation history if needed for next prediction
                if step >= self.Ta - self.To + 1:
                    self.episode_manager.update_observation(
                        state_data['policy_side_img'] / 255.0,
                        state_data['policy_wrist_img'] / 255.0,
                        state_data['tcp_pose'] if self.state_type == 'ee_pose' else state_data['joint_pos']
                    )
                
                # Ensure we keep the desired framerate
                time.sleep(max(1 / 10 - (time.time() - start_time), 0))
                j += 1
            
            # Store predicted absolute actions for rewinding
            self._last_predicted_abs_actions = predicted_abs_actions
            
            # Process data for failure detection module
            if self.failure_detection_module:
                step_data = {
                    'step_type': 'policy_step',
                    'action_seq': action_seq,
                    'predicted_abs_actions': predicted_abs_actions,
                    'policy_obs': policy_obs,
                    'curr_latent': curr_latent,
                    'timestep': j,
                    'episode_manager': self.episode_manager
                }
                
                failure_step_data = self.failure_detection_module.process_step(step_data)
                
                failure_flag, failure_reason = self.failure_detection_module.detect_failure(
                    timestep=j,
                    max_episode_length=self.max_episode_length,
                    failure_step_data=failure_step_data
                )
                
                if failure_flag or j >= self.max_episode_length - self.Ta:
                    if failure_flag:
                        print(f"Failure detected: {failure_reason}")
                    else:
                        print("Maximum episode length reached")
                    
                    # Send human check request
                    if failure_flag:
                        self.socket.send(f"NEED_HUMAN_CHECK_from_robot{self.robot_id}_for_failure")
                    else:
                        self.socket.send(f"NEED_HUMAN_CHECK_from_robot{self.robot_id}_for_timeout")
                    
                    # Pause policy execution
                    self.robot_state = "idle"
                    break
            
            # Check for maximum episode length without failure detection
            elif j >= self.max_episode_length - self.Ta:
                print("Maximum episode length reached")
                self.socket.send(f"NEED_HUMAN_CHECK_from_robot{self.robot_id}_for_timeout")
                self.robot_state = "idle"
                break
            
            # Add current state to demo buffer for possible rewinding
            if j % self.Ta == 0:
                episode_data = {}
                for k, v in episode_buffers.items():
                    if len(v) > 0:
                        episode_data[k] = v[-self.Ta:]
                
                if len(self.demo_buffer) <= j // self.Ta:
                    self.demo_buffer.append(episode_data)
                else:
                    self.demo_buffer[j // self.Ta] = episode_data
        
        return episode_buffers
    
    def rewind_robot(self, steps=3):
        """Rewind robot by a number of Ta-step chunks"""
        if not hasattr(self, '_last_predicted_abs_actions') or self._last_predicted_abs_actions is None:
            print("No previous actions to rewind from")
            return
        
        print(f"Rewinding robot by {steps} Ta-step chunks...")
        
        # Calculate how many steps we can rewind
        max_rewind_steps = min(steps * self.Ta, len(self.demo_buffer) * self.Ta)
        rewind_steps = 0
        
        # Current robot pose
        curr_tcp = self._last_predicted_abs_actions[0, self.Ta-1, :7]
        curr_pos = curr_tcp[:3]
        curr_rot = R.from_quat(curr_tcp[3:], scalar_first=True)
        
        # Rewind loop
        for i in range(max_rewind_steps):
            if i % self.Ta == 0 and i > 0:
                print(f"Rewound {i//self.Ta} Ta-step chunks")
                
            # Get the most recent demo buffer
            buffer_idx = len(self.demo_buffer) - 1 - (i // self.Ta)
            step_idx = self.Ta - 1 - (i % self.Ta)
            
            if buffer_idx < 0 or step_idx < 0:
                break
                
            if len(self.demo_buffer) <= buffer_idx:
                break
                
            if len(self.demo_buffer[buffer_idx]['action']) <= step_idx:
                break
                
            # Get previous action to rewind
            prev_action = self.demo_buffer[buffer_idx]['action'][step_idx]
            
            # Rewind robot by applying inverse action
            curr_pos, curr_rot = self.robot_env.rewind_robot(curr_pos, curr_rot, prev_action)
            rewind_steps += 1
        
        print(f"Successfully rewound {rewind_steps} steps")
        return curr_pos, curr_rot
    
    def detach(self):
        """
        Detach sigma device from robot. 
        1. The design philosophy of detach is to allow sigma to move freely while the robot stays still
        2. Detach must be called at the latest moment when the robot arm is not being controlled
        3. At the beginning of teleop, detach can also be used to keep the robot still
        """
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
    
    def run(self):
        """Main run loop"""
        while self.running:
            if self.robot_state == "agent_controlled":
                episode_buffers = self.run_policy()
                # Save episode data
                if episode_buffers:
                    self._save_episode_data(episode_buffers)
            time.sleep(0.1)
    
    def _save_episode_data(self, episode_buffers):
        """Save episode data to replay buffer"""
        # Implementation for saving to buffer
        episode = {}
        
        for key, value in episode_buffers.items():
            if len(value) > 0:
                episode[key] = np.stack(value, axis=0)
        
        # Handle action mode post-processing if needed
        if hasattr(self.config, 'post_process_action_mode') and self.config.post_process_action_mode:
            episode['action_mode'] = postprocess_action_mode(np.array(episode_buffers['action_mode']))
        else:
            episode['action_mode'] = np.array(episode_buffers['action_mode'])
            
        # Save to disk or memory as needed
        print(f"Episode {self.episode_idx} completed with {len(episode_buffers['action'])} steps")
        self.episode_idx += 1