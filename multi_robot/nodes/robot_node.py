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
from multi_robot.utils.message_distillation import parse_message_regex, split_combined_messages, MessageHandler
from diffusion_policy.diffusion_policy.env_runner.real_robot_runner import RealRobotRunner


class RobotNode(RealRobotRunner):
    """Robot node that manages policy execution, teleoperation, and human intervention.
    Handles autonomous control, human takeover, and communication with teleop nodes."""
    def __init__(self, config, rank: int, device_ids: List[int]):
        # Configuration and device setup for multi-robot
        self._setup_robot_config(config, rank, device_ids)
        
        # Initialize MessageHandler as a component first
        from multi_robot.utils.message_distillation import MessageHandler
        self.message_handler = MessageHandler()
        self._setup_message_routes()
        
        # Initialize base RealRobotRunner with adapted config
        eval_cfg = self._adapt_config_for_base(config)
        super().__init__(eval_cfg, rank, device_ids)
        
        # Initialize communication
        self._setup_communication()
        
        # Initialize robot-specific state management
        self._initialize_robot_state()

    def _setup_robot_config(self, config, rank: int, device_ids: List[int]):
        """Setup robot-specific configuration and device parameters."""
        self.config = config
        # Robot-specific configuration
        self.robot_id = config.robot_info.robot_id
        self.is_multi_robot_env = config.robot_info.num_robots > 1
        self.robot_info_dict = config.robot_info.robot_info_dict
    
    def _adapt_config_for_base(self, config):
        """Adapt robot config to work with RealRobotRunner base class."""
        # Create a compatible eval_cfg for the base class
        eval_cfg = type('EvalConfig', (), {})()
        
        # Copy essential attributes
        eval_cfg.checkpoint_path = config.checkpoint_path
        eval_cfg.policy = config.policy if hasattr(config, 'policy') else type('Policy', (), {'num_inference_steps': 1})()
        eval_cfg.output_dir = config.output_dir
        eval_cfg.train_dataset_path = config.train_dataset_path
        eval_cfg.save_buffer_path = config.save_buffer_path
        eval_cfg.seed = config.seed
        eval_cfg.failure_detection_module = getattr(config, 'failure_detection_module', None)
        eval_cfg.random_init = getattr(config, 'random_init', False)
        eval_cfg.post_process_action_mode = getattr(config, 'post_process_action_mode', False)
        eval_cfg.sirius_termination = getattr(config, 'sirius_termination', False)
        eval_cfg.num_samples = getattr(config, 'num_samples', 1)
        eval_cfg.Ta = getattr(config, 'Ta', None)
        
        return eval_cfg

    def _setup_communication(self):
        """Setup communication with hub."""
        socket_ip = self.config.robot_info.socket_ip
        socket_port = self.config.robot_info.socket_port
        self.socket = SocketClient(socket_ip, socket_port, message_handler=self.message_handler.handle_message)
        self.socket.start_connection()
        self.lock = threading.Lock()

    def _initialize_robot_env(self):
        """Override to support multi-robot environment initialization."""
        self.robot_env = RobotEnv(
            camera_serial=CAM_SERIAL if not self.is_multi_robot_env else self.config.camera.serial, 
            img_shape=self.img_shape, 
            fps=self.fps,
            is_multi_robot_env=True,
            robot_id=self.robot_id,
            robot_info_dict=self.robot_info_dict
        )

    def _initialize_robot_state(self):
        """Initialize robot-specific state variables."""
        # Robot state management
        self.robot_state = "idle"  # idle / teleop_controlled / agent_controlled / error
        self.teleop_id = 0
        self.last_query_robot = time.time() - 1
        self.inform_freq = 5  # Hz
        self.teleop_ctrl_freq = 10  # Hz
        
        # Running state
        self.running = True
        
        # Sigma device control variables
        self._initialize_sigma_variables()
        
        # Episode and scene alignment state
        self.finish_episode = False
        self.scene_alignment_completed = False

    def _initialize_sigma_variables(self):
        """Initialize sigma device control variables."""
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
        self.rewind_key = False
        self.rewind_pos = None
        self.rewind_rot = None
        self.ready_to_stop_flag = True
        self.stop_event = None
        self.stop_teleop_key = False
        self._last_predicted_abs_actions = None

    def _setup_message_routes(self):
        """Define message routing table for different message types and their handlers.
        Maps incoming messages to appropriate handler methods."""
        handlers = {
            "READY": self.get_ready_takeover,
            "TELEOP_TAKEOVER_RESULT": self._handle_teleop_takeover_result,
            "CONTINUE_POLICY": self.handle_ctn,
            "TELEOP_CTRL_START": self.start_being_teleoped,
            "COMMAND": self.process_teleop_command,
            "TELEOP_CTRL_STOP": self.ready_to_stop,
            "THROTTLE_SHIFT": self.process_throttle_info,
            "REWIND_ROBOT": self._handle_rewind_robot,
            "SCENE_ALIGNMENT_COMPLETED": self.handle_scene_alignment_completed,
        }
        for msg_type, handler in handlers.items():
            self.message_handler.register_handler(msg_type, handler)

    def _handle_teleop_takeover_result(self, message: str):
        """Process the result of a teleop takeover request.
        Handles both success and failure cases from human operator."""
        self.detach()
        if "SUCCESS" in message:
            self.stop_event = "accept"
            if self.failure_reason is not None and self.failure_reason != "maximum episode length reached":
                self.handle_redundant_failure()
        else:
            self.stop_event = "cancel"
        with self.lock:
            self.finish_episode = True  #finish_episode has greater priority over robot_state

    def _handle_rewind_robot(self, message: str):
        """Handle robot rewind request from teleop.
        Sets flag to initiate rewinding process."""
        self.rewind_key = True

    def _handle_unknown_message(self, message: str):
        """Handle unknown message types.
        Logs unrecognized messages for debugging."""
        print(f"Unknown command: {message}")
    
        

    
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
    


    def _run_rollout(self):
        """Main rollout loop"""
        try:
            while True: 
                print(f"Rollout episode: {self.episode_idx}")
                
                # Run single episode
                episode_data = self._run_single_episode()
                
                if episode_data is not None:
                    # Save episode to replay buffer
                    self.replay_buffer.add_episode(episode_data, compressors='disk')
                    self.episode_id = self.replay_buffer.n_episodes - 1
                    print(f'Saved episode {self.episode_id}')
                
                # Reset robot between episodes
                self.reset()  
                self.failure_detection_module.failure_detector.last_predicted_abs_actions = None
                
                self.episode_idx += 1
                
                # Check termination conditions
                if self._should_terminate():
                    break
                
                # For scene configuration reset
                time.sleep(5)
        
        finally:
            self._cleanup()

    def _run_single_episode(self):
        """Run a single episode of policy execution or teleoperation.
        Handles initialization, execution, and data collection for one episode."""
        time.sleep(1)

        # Reset keyboard states
        with self.lock:
            self.finish_episode = False
        self.robot_state = "agent_controlled"  
        self.stop_event = None

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
                self.run_policy() 
                
            elif self.robot_state == "teleop_controlled":
                time.sleep(0.1)  # Prevent high CPU usage
            elif self.robot_state == "idle":
                time.sleep(0.1)
                if self.rewind_key:
                    self.handle_prerewind_robot()
                    self.rewind_key = False

            with self.lock:
                finish_episode_flag = self.finish_episode
            
            if finish_episode_flag:
                break

        # Finalize episode and return data
        if self.finish_episode and self.stop_event == "accept":
            episode_data = self._finalize_episode(self.episode_buffers, self.episode_id)
            return episode_data
        else:
            return None
            
    
    def run_policy(self):
        """Run policy inference loop"""
        while self.j < self.max_episode_length and self.robot_state == "agent_controlled" : 
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
                    if self.failure_detection_module and hasattr(self.failure_detection_module, 'should_stop_rewinding') and self.failure_detection_module.enable_OT:
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
                
                failure_step_data = self.failure_detection_module.process_step(step_data)
                
                failure_flag, self.failure_reason = self.failure_detection_module.detect_failure( 
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
                self.call_human_for_help("timeout")
                return  
            
    def call_human_for_help(self,reason):
        """Request human assistance when needed.
        Sends message to hub requesting human check with specified reason."""
        self.socket.send(f"NEED_HUMAN_CHECK_from_robot{self.robot_id}_for_{reason}") 
    
    def handle_ctn(self):
        """Handle continue policy command from teleop.
        Resumes autonomous policy execution after human check."""
        self.robot_state = "agent_controlled"
        if self.j >= self.max_episode_length - self.Ta: # after hum
            print("Maximum episode length reached, turning to human for help.")
            self.call_human_for_help("timeout")
            self.robot_state = "idle"

        print("False Positive failure! Continue policy rollout.")
        if self.failure_reason == 'action inconsistency' and self.failure_detection_module.enable_action_inconsistency:
            self.failure_detection_module.failure_detector.expert_action_threshold = np.inf
            print("Reset the action inconsistency threshold to infinity temporarily")
    
    def handle_redundant_failure(self):
        """Handle redundant failure detection.
        Removes redundant failure logs when human intervention succeeds.""" 
        if hasattr(self.failure_detection_module, 'failure_logs'):
            if self.failure_detection_module.failure_logs:  # Check if dictionary is not empty
                self.failure_detection_module.failure_logs.popitem()
        elif hasattr(self.failure_detection_module, 'failure_indices'):
            if self.failure_detection_module.failure_indices:  # Check if list is not empty
                self.failure_detection_module.failure_indices.pop()
                    
    
    def get_ready_takeover(self, message):
        """Prepare for human takeover.
        Extracts teleop ID and prepares for teleoperation."""
        """Prepare for takeover"""
        templ = "READY_for_state_check_by_human_with_teleop_id_{}"
        teleop_id = parse_message_regex(message, templ)[0]
        self.teleop_id = teleop_id

    def process_throttle_info(self, msg):
        """Process throttle shift information from teleop.
        Parses position and rotation deltas for teleoperation."""
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
        self.robot_state = "teleop_controlled"
    
    def reset_teleop_cmd(self):
        """Reset teleop command variables.
        Clears all teleoperation state variables after session ends."""
        """Reset teleop command variables"""
        self.last_p = None
        self.last_r = None
        self.diff_p_arr = None
        self.diff_r_arr = None
        self.width = None
        self.throttle = None
        self.delta_p_arr = None
        self.delta_r_arr = None
        self.abs_p = None
        self.abs_r =None
        self.curr_p_action = None
        self.curr_r_action =None
    
    def handle_prerewind_robot(self):
        """Handle pre-rewind robot state.
        Performs rewinding process and notifies teleop of completion."""
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
        """Request scene alignment from teleop node.
        Initiates scene alignment process with context information."""
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
        """Handle scene alignment completion notification.
        Updates scene alignment state when teleop confirms completion."""
        """Handle scene alignment completion notification"""
        self.scene_alignment_completed = True
    
    def request_scene_alignment_with_reference(self, ref_side_cam, ref_wrist_cam):
        """Request scene alignment with reference images.
        Sends reference images to teleop for alignment guidance.""" 
        print("Requesting scene alignment with reference images from teleop") 
        # Send simple scene alignment request to teleop (no image data)
        align_msg = f"SCENE_ALIGNMENT_WITH_REF_REQUEST_{self.robot_id}_rewind"
        self.socket.send(align_msg)
        
        # Start local image display with reference images in robot end
        ref_side_img = cv2.cvtColor(ref_side_cam, cv2.COLOR_RGB2BGR)
        ref_wrist_img = cv2.cvtColor(ref_wrist_cam, cv2.COLOR_RGB2BGR)
        self.start_scene_alignment_display_with_reference(ref_side_img, ref_wrist_img)  
        
        print("Scene alignment with reference completed")
    
    def start_scene_alignment_display(self, episode_idx):
        """Start scene alignment display with saved reference images.
        Loads and displays reference images for scene alignment."""
        """Start scene alignment display in robot end, wait for teleop confirmation"""
        # Load reference images from saved files
        ref_side_img = cv2.imread(f"{self.output_dir}/side_{episode_idx}.png")
        ref_wrist_img = cv2.imread(f"{self.output_dir}/wrist_{episode_idx}.png")

        # Use the restored local scene alignment display
        if ref_side_img is not None and ref_wrist_img is not None:
            self.start_scene_alignment_display_with_reference(ref_side_img, ref_wrist_img, raw=True)
        else:
            print(f"Could not load reference images for episode {episode_idx}")
            print("Proceeding without scene alignment...")
            time.sleep(2.0)  # Brief pause to allow manual alignment
    
    def start_scene_alignment_display_with_reference(self, ref_side_img, ref_wrist_img, raw=False):
        """Start scene alignment display with provided reference images"""
        print("Scene alignment display with reference started - waiting for teleop confirmation")  
        cv2.namedWindow("Wrist", cv2.WINDOW_AUTOSIZE)

        print("=================start image display=================")
        
        # Reset completion flag
        self.scene_alignment_completed = False
        
        # Display images until teleop confirms completion or timeout
        while not self.scene_alignment_completed:
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
                print(f"Error during scene alignment display: {e}")
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
        self.send_resume_sigma() 
        self.send_transform_sigma(translate, rotation)
        self.robot_state = "teleop_controlled"
        self.ready_to_stop_flag = False
        self.stop_teleop_key = False
        self.teleop_thread_running = True
        self.teleop_thread = threading.Thread(target=self.teleop_thread_loop, daemon=True)
        self.teleop_thread.start()
        
    
    def ready_to_stop(self,message):
        """Process ready-to-stop signal from teleop.
        Sets stop event type based on teleop message."""
        self.ready_to_stop_flag = True
        if "cancel" in message:
            self.stop_event = "cancel"
        elif "accept" in message:
            self.stop_event = "accept"
        elif "continue" in message:
            self.stop_event = "continue"

    def stop_teleop(self):
        """Stop teleoperation mode.
        Handles transition from teleoperation back to autonomous control."""
        self.teleop_thread_running = False
        self.teleop_thread.join()
        self.teleop_thread = None
        self.robot_state = "idle"
        print("Teleoperation terminated, state switched to 'AGENT_CONTROLLED'.")
        
        # Reset pose for policy after human intervention
        self.episode_manager.initialize_pose(self.last_p, self.last_r.as_quat(scalar_first=True))

        self.reset_teleop_cmd()
        self.detach()
        assert self.stop_event in ["cancel","accept","continue"]

        if self.stop_event != "continue":  #success or failure
            with self.lock:
                self.finish_episode = True   #finish_episode has greater priority over robot_state
        else:
            self.robot_state = "agent_controlled"
            self.handle_ctn()
    
    def process_teleop_command(self, message):
        """Process teleoperation command from teleop node.
        Parses and executes teleop commands on the robot.""" 
        if "sigma" in message and not self.stop_teleop_key :
            pattern = r"COMMAND_from_(\d+)_to_(\d+):sigma:\[([^\]]+)\],\[([^\]]+)\],([^,]+),([^,]+)"
            match = re.match(pattern, message)
            
            if not match:
                raise ValueError(f"Invalid command format: {message[:100]}{'...' if len(message) > 100 else ''}")
                
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
            except ValueError as e:
                raise ValueError(f"Failed to parse numeric values from command: {e}")
                        
    def inform_robot_state(self):
        """Periodically report robot state"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_query_robot >= 1.0 / self.inform_freq:
                self.last_query_robot = current_time
                self.socket.send(f"INFORM_ROBOT_STATE_{self.robot_id}_{self.robot_state}")
            time.sleep(0.01)
    
    def teleop_thread_loop(self):
        while self.teleop_thread_running:
            self.teleop_action_step()
    
    def teleop_action_step(self):
        """Execute a single teleop action step"""
        start_time = time.time()

        #Get camera data and robot state
        state_data = self.robot_env.get_robot_state()
        tcp_pose = state_data['tcp_pose']
        joint_pos = state_data['joint_pos']

        # Get teleop controls
        self.abs_p = self.robot_env.robot.init_pose[:3] + self.diff_p_arr
        self.abs_r = R.from_quat(self.robot_env.robot.init_pose[3:7], scalar_first=True).inv() * R.from_quat(self.diff_r_arr, scalar_first=True)
        self.curr_p_action = self.abs_p - self.last_p
        self.curr_r_action = self.last_r.inv() * self.abs_r
        self.last_p = self.abs_p    
        self.last_r = self.abs_r    

        if self.diff_p_arr is None or self.diff_r_arr is None:
            return

        # Handle throttle detach
        if self.throttle < -0.9:
            if not self.last_throttle:
                self.detach()
                self.last_throttle = True
            return

        if self.last_throttle:
            self.last_throttle = False
            self.send_resume_sigma(during_teleop=True)  # "during_teleop" is to make sure the action recorded is right after the detach
            if self.delta_p_arr is None or self.delta_r_arr is None: 
                self.robot_state = "idle"
            else:
                self.last_p = self.delta_p_arr + self.robot_env.robot.init_pose[:3]
                self.last_r = R.from_quat(self.robot_env.robot.init_pose[3:7], scalar_first=True) * R.from_quat(self.delta_r_arr, scalar_first=True)
                self.delta_p_arr = None
                self.delta_r_arr = None
            return

        # Execute a single teleop action step
        if self.robot_state == "teleop_controlled": 
            # Deploy action to robot
            self.robot_env.deploy_action(
                np.concatenate((self.abs_p, self.abs_r.as_quat(scalar_first=True))),
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
                'action': np.concatenate((self.curr_p_action, self.curr_r_action.as_quat(scalar_first=True), [gripper_action])),
                'action_mode': INTV
            }
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

             # Sleep to maintain fps
            time.sleep(max(1 / self.teleop_ctrl_freq - (time.time() - start_time), 0))

        # Check if ready to stop
        if self.ready_to_stop_flag and self.j % self.Ta == 0:
            self.stop_teleop_key = True
            self.stop_teleop()
    
    def reset(self,random_init=False, random_init_pose=None):
        """Reset robot to initial state.
        Optionally uses random initialization for exploration."""
        state=self.robot_env.reset_robot(random_init, random_init_pose)
        self.send_reset_sigma()
        tcp = state["tcp_pose"]
        translate = tcp[0:3] - self.robot_env.home_pose[0:3]
        rotation = (R.from_quat(self.robot_env.home_pose[3:], scalar_first=True).inv() * R.from_quat(tcp[3:],scalar_first=True)).as_quat(scalar_first=True)
        self.send_transform_sigma(translate,rotation)
        return state
        
    def detach(self):
        """Detach sigma device from robot control.
        Records current TCP position and sends detach command."""
        self.detach_tcp = self.robot_env.robot.get_tcp_pose()
        print(f"Detaching sigma at TCP position: {self.detach_tcp}")
        self.send_detach_sigma()
    
    def send_transform_sigma(self, translate, rotation):
        """Send transform message to sigma device.
        Transmits transformation data to align haptic device with robot.""" 
        msg = f"SIGMA_TRANSFORM_from_{self.robot_id}_{translate.tolist() + rotation.tolist()}_to_{self.teleop_id}".encode()
        self.socket.send(msg)
    
    def send_detach_sigma(self):
        """Send detach message to sigma device.
        Instructs teleop to detach haptic device from robot control."""
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
    

    
    def _main_thread(self):
        """Main execution thread.Starts state reporting thread and runs rollout loop."""
         # Start state report thread
        self.inform_thread = threading.Thread(target=self.inform_robot_state, daemon=True)
        self.inform_thread.start()

        self._run_rollout()
    
    