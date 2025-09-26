import os
import sys
import re
import time
import numpy as np
import torch
import threading
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R
import cv2
from omegaconf import DictConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from hardware.robot_env import RobotEnv, ROBOT, INTV
from armada.communication.socket_client import SocketClient
from armada.utils.message_distillation import parse_message_regex
from armada.env_runner.real_env_runner import RealEnvRunner
from armada.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply


class RobotNode(RealEnvRunner):
    """Robot node for paralleled autonomous policy rollout, based on single-robot env runner"""
    def __init__(self, 
                 cfg: DictConfig, 
                 rank: int, 
                 device_ids: List[int]):
        super().__init__(cfg, rank, device_ids)
        
        self.robot_id = self.cfg.robot_info.robot_id

        # Initialize MessageHandler as a component first
        from armada.utils.message_distillation import MessageHandler
        self.message_handler = MessageHandler()
        self._setup_message_routes()
        
        # Initialize communication
        self._setup_communication()
        
        # Initialize robot-specific state management
        self._initialize_robot_state()

    def _setup_communication(self):
        """Setup communication with hub."""
        socket_ip = self.cfg.robot_info.socket_ip
        socket_port = self.cfg.robot_info.socket_port
        self.socket = SocketClient(socket_ip, socket_port, message_handler=self.message_handler.handle_message)
        self.socket.start_connection()
        self.lock = threading.Lock()

    def _initialize_robot_env(self):
        """Override to support multi-robot environment initialization."""
        self.robot_env = RobotEnv(
            camera_serial=self.cfg.camera.serial, 
            img_shape=self.img_shape, 
            fps=self.fps,
            is_multi_robot_env=True,
            robot_id=self.cfg.robot_info.robot_id,
            robot_info_dict=self.cfg.robot_info.robot_info_dict
        )

    def _initialize_robot_state(self):
        """Initialize robot-specific state variables."""
        # Robot state management
        self.robot_state = "idle"  # idle / teleop_controlled / agent_controlled / error
        self.teleop_id = 0
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
        self.latest_command_time = None
        self.rewind_key = False
        self.rewind_pos = None
        self.rewind_rot = None
        self.ready_to_stop_flag = True
        self.stop_event = None

    def _setup_message_routes(self):
        """Define message routing table for different message types and their handlers.
        Maps incoming messages to appropriate handler methods."""
        handlers = {
            "READY": self._get_ready_takeover,
            "TELEOP_TAKEOVER_RESULT": self._handle_teleop_takeover_result,
            "CONTINUE_POLICY": self._handle_ctn,
            "TELEOP_CTRL_START": self._start_being_teleoped,
            "COMMAND": self._process_teleop_command,
            "TELEOP_CTRL_STOP": self._ready_to_stop,
            "REWIND_ROBOT": self._handle_rewind_robot,
            "SCENE_ALIGNMENT_COMPLETED": self._handle_scene_alignment_completed,
            "QUIT": self._handle_quit,
        }
        for msg_type, handler in handlers.items():
            self.message_handler.register_handler(msg_type, handler)

    def _handle_teleop_takeover_result(self, message: str):
        """Process the result of a teleop takeover request.
        Handles both success and failure cases from human operator."""
        self.detach()
        if "SUCCESS" in message:
            self.stop_event = "accept"
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
    
    def _handle_quit(self, message: str):
        """Handle quit command from teleop.
        Instructs robot to quit."""
        self.stop_event = "quit"
        self.running = False
        with self.lock:
            self.finish_episode = True

    def run_rollout(self):
        """Main rollout loop"""
        self.inform_robot_state()
        try:
            while self.running: 
                
                print(f"Rollout episode: {self.episode_idx}")
                
                # Run single episode
                episode_data = self._run_single_episode()
                
                if episode_data is not None:
                    # Save episode to replay buffer
                    self.replay_buffer.add_episode(episode_data, compressors='disk')
                    self.saved_episode_idx = self.replay_buffer.n_episodes - 1
                    print(f'Saved episode {self.saved_episode_idx}')
                
                # Reset robot between episodes
                self.reset()  
                
                self.episode_idx += 1
                
                # For scene configuration reset
                time.sleep(5)
        
        finally:
            self._cleanup()

    def _run_single_episode(self) -> Optional[Dict[str, Any]]:
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
        if getattr(self.cfg, 'random_init', False):
            random_init_pose = self.robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
        
        robot_state = self.reset(getattr(self.cfg, 'random_init', False), random_init_pose)
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
        if getattr(self.cfg, 'random_init', False) and random_init_pose is not None:
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
            init_policy_obs = self.episode_manager.get_policy_observation() # Keep dim but only require the first sample
            for key, value in init_policy_obs.items():
                init_policy_obs[key] = value[0:1]
            with torch.no_grad():
                init_latent = self.policy.extract_latent(init_policy_obs)
                init_latent = init_latent.reshape(-1)
            self.failure_detection_module.process_step({
                'step_type': 'episode_start',
                'episode_idx': self.episode_idx,
                'rollout_init_latent': init_latent.unsqueeze(0) # For determining expert candidates
            })

        # Policy inference loop
        self.j = 0  # Episode timestep     

        while self.running:
            if self.j >= self.max_episode_length and self.robot_state == "agent_controlled":
                self.stop_event = "timeout"
                self.call_timeout()
                self.finish_episode = True

            # Handle robot state actions
            if self.robot_state == "agent_controlled":
                self._run_policy_inference_loop() 
                
            elif self.robot_state == "teleop_controlled":
                time.sleep(0.1)  # Prevent high CPU usage
            elif self.robot_state == "idle":
                time.sleep(0.1)
                if self.rewind_key:
                    self._rewind_robot()
                    self.rewind_key = False

            with self.lock:
                finish_episode_flag = self.finish_episode
            
            if finish_episode_flag:
                break

        # Finalize episode and return data
        if self.finish_episode and (self.stop_event == "accept" or self.stop_event == "quit" or self.stop_event == "timeout"):
            episode_data = self._finalize_episode()
            return episode_data
        else:
            return None
            
    
    def _run_policy_inference_loop(self):
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
                if self.failure_detection_module and hasattr(self.failure_detection_module, 'should_stop_rewinding'):
                    curr_action, curr_latent = self.policy.predict_action(policy_obs, return_latent=True)
                else:
                    curr_action = self.policy.predict_action(policy_obs)
                    curr_latent = None
            
            # Get first Ta actions and execute on robot
            np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
            action_seq = np_action_dict['action']
            
            # Execute action sequence
            for step in range(self.Ta):
                if step > 0:
                    start_time = time.time()
                
                # Get robot state
                state_data = robot_state if step == 0 else self.robot_env.get_robot_state()
                
                # Get absolute action for this step
                deployed_action, gripper_action, curr_p, curr_r, curr_p_action, curr_r_action = \
                    self.episode_manager.get_absolute_action_for_step(action_seq, step)
                
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
            
            # ================Detect failure===============
            if self.failure_detection_module:
                step_data = {
                    'step_type': 'policy_step',
                    'curr_latent': curr_latent,
                    'timestep': self.j,
                    'robot_state': robot_state
                }
                
                self.failure_detection_module.process_step(step_data)
                
                failure_flag, failure_reason, _ = self.failure_detection_module.detect_failure(
                    timestep=self.j,
                    max_episode_length=self.max_episode_length
                )

                if self.j >= self.max_episode_length: # Making sure that every failure detection result is processed, raising recall
                    failure_flag, failure_reason, _ = self.failure_detection_module.wait_for_final_results(self.j)

                print(f"=========== Global timestep: {self.j // self.Ta - 1} =============")

                if failure_flag or self.j >= self.max_episode_length:
                    self.robot_state = "idle"         #temporarily set to idle for human judgement

                    if failure_flag:
                        print(f"Failure detected! Due to {failure_reason}")
                    else:
                        print("Maximum episode length reached")

                     # Send human check request
                    if failure_flag:
                        self.call_human_for_help("failure")
                    else:
                        self.stop_event = "timeout"
                        self.call_timeout()
                        self.finish_episode = True

                    return  
            # ================Detect failure end===============
            
            # Check for maximum episode length without failure detection
            elif self.j >= self.max_episode_length:
                self.robot_state = "idle"
                print("Maximum episode length reached")
                self.stop_event = "timeout"
                self.call_timeout()
                self.finish_episode = True
                return  
            
    def call_human_for_help(self,reason):
        """Request human assistance when needed.
        Sends message to hub requesting human check with specified reason."""
        self.socket.send(f"NEED_HUMAN_CHECK_from_robot{self.robot_id}_for_{reason}") 
    
    def call_timeout(self):
        self.socket.send(f"TIMEOUT_of_{self.robot_id}") 
    
    def _handle_ctn(self, message=None):
        """Handle continue policy command from teleop.
        Resumes autonomous policy execution after human check."""
        self.robot_state = "agent_controlled"
        if self.j >= self.max_episode_length: # after human decide
            print("Maximum episode length reached, turning to human for help.")
            self.stop_event = "cancel"
            self.call_timeout()
            self.finish_episode = True

        print("False Positive failure! Continue policy rollout.")
    
    def _get_ready_takeover(self, message):
        """Prepare for human takeover.
        Extracts teleop ID and prepares for teleoperation."""
        """Prepare for takeover"""
        templ = "READY_for_state_check_by_human_with_teleop_id_{}"
        teleop_id = parse_message_regex(message, templ)[0]
        self.teleop_id = teleop_id
    
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
        self.latest_command_time = None
    
    def _rewind_robot(self):
        """Handle pre-rewind robot state.
        Performs rewinding process and notifies teleop of completion."""
        # Perform rewinding if needed
        print("Starting rewinding process...")
        try:
            # Use the last target action for rewinding
            curr_pos = self.episode_manager.last_p[0]
            curr_rot = self.episode_manager.last_r[0]
            
            # Let failure detection module determine rewinding behavior
            if self.failure_detection_module and hasattr(self.failure_detection_module, 'should_stop_rewinding'):
                prev_side_cam, prev_wrist_cam, curr_pos, curr_rot = self._rewind_with_failure_detection(curr_pos, curr_rot)
            else:
                prev_side_cam, prev_wrist_cam, curr_pos, curr_rot = self._rewind_simple(curr_pos, curr_rot)
            
            if "prev_side_cam" in locals():
                print("Please reset the scene and press 'c' in teleop node to go on to human intervention")
                self.request_scene_alignment_with_reference(prev_side_cam, prev_wrist_cam)
            
            self.rewind_pos, self.rewind_rot = curr_pos, curr_rot
            print(f"Rewind completed. New timestep: {self.j}")
        except Exception as e:
            print(f"Error during rewind: {e}")
            print("Continuing without rewind...")
    
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
    
    def _handle_scene_alignment_completed(self, message):
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
          
    
    def _start_being_teleoped(self, message):
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
        self.teleop_thread = threading.Thread(target=self.teleop_thread_loop, daemon=True)
        self.teleop_thread.start()
        
    
    def _ready_to_stop(self,message):
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
            self._handle_ctn()
    
    def _process_teleop_command(self, message):
        """Process teleoperation command from teleop node.
        Parses and executes teleop commands on the robot.""" 
        if "sigma" in message:
            pattern = r"COMMAND_from_(\d+)_to_(\d+):sigma:\[([^\]]+)\],\[([^\]]+)\],([^,]+),([^,]+),([^,]+)"
            match = re.match(pattern, message)
            
            if not match:
                raise ValueError(f"Invalid command format: {message[:100]}{'...' if len(message) > 100 else ''}")
                
            teleop_id = match.group(1)
            rbt_id = match.group(2)
            diff_p_str = match.group(3)
            diff_r_str = match.group(4)
            width = match.group(5)
            throttle = match.group(6)
            curr_time = match.group(7)            
            try:
                # Parse arrays with better error handling
                self.throttle = float(throttle)
                if self.throttle >= -0.9:
                    self.diff_p_arr = np.array([float(x.strip()) for x in diff_p_str.split(",")]) 
                    self.diff_r_arr = np.array([float(x.strip()) for x in diff_r_str.split(",")]) 
                    self.width = float(width)  
                    self.latest_command_time = float(curr_time)
            except ValueError as e:
                raise ValueError(f"Failed to parse numeric values from command: {e}")
                        
    def inform_robot_state(self):
        """Report robot state to the communication hub"""
        self.socket.send(f"INFORM_ROBOT_STATE_{self.robot_id}_{self.robot_state}")
    
    def teleop_thread_loop(self):
        while not self.ready_to_stop_flag or self.robot_state == 'teleop_controlled':
            self.teleop_action_step()
    
    def teleop_action_step(self):
        """Execute a single teleop action step"""
        start_time = time.time()

        #Get camera data and robot state
        state_data = self.robot_env.get_robot_state()

        # Get teleop controls
        if self.diff_p_arr is None or self.diff_r_arr is None or time.time() - self.latest_command_time > 0.1: # Closed-loop command filtering
            time.sleep(1 / self.teleop_ctrl_freq - (time.time() - start_time))
            return
    
        abs_p = self.robot_env.robot.init_pose[:3] + self.diff_p_arr
        abs_r = R.from_quat(self.robot_env.robot.init_pose[3:7], scalar_first=True) * R.from_quat(self.diff_r_arr, scalar_first=True)
        curr_p_action = abs_p - self.last_p
        curr_r_action = self.last_r.inv() * abs_r
        self.last_p = abs_p
        self.last_r = abs_r

        # Handle throttle detach
        if self.throttle < -0.9:
            if not self.last_throttle:
                self.record_detach_tcp()
                self.last_throttle = True
            return

        if self.last_throttle:
            self.last_throttle = False
            self.last_p = state_data['tcp_pose'][:3]
            self.last_r = R.from_quat(state_data['tcp_pose'][3:], scalar_first=True)
            return

        # Execute a single teleop action step
        if self.robot_state == "teleop_controlled": 
            # Deploy action to robot
            gripper_action = self.robot_env.gripper.max_width * self.width / 1000
            self.robot_env.deploy_action(
                np.concatenate((abs_p, abs_r.as_quat(scalar_first=True))),
                gripper_action
            )
            
            # Save demo data for return
            self.episode_manager.update_observation(
                state_data['policy_side_img'] / 255.0,
                state_data['policy_wrist_img'] / 255.0,
                state_data['tcp_pose'] if self.state_type == 'ee_pose' else state_data['joint_pos']
            )
            
            # Store demo data
            self.episode_buffers['wrist_cam'].append(state_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            self.episode_buffers['side_cam'].append(state_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            self.episode_buffers['tcp_pose'].append(state_data['tcp_pose'])
            self.episode_buffers['joint_pos'].append(state_data['joint_pos'])
            self.episode_buffers['action'].append(np.concatenate((curr_p_action, curr_r_action.as_quat(scalar_first=True), [gripper_action])))
            self.episode_buffers['action_mode'].append(INTV)
            self.j += 1

             # Sleep to maintain fps
            time.sleep(max(1 / self.teleop_ctrl_freq - (time.time() - start_time), 0))

        # Check if ready to stop
        if self.ready_to_stop_flag:
            while self.j % self.Ta != 0: # Fill with dummy actions (do nothing after the task is finished)
                curr_p_action = abs_p - self.last_p
                curr_r_action = self.last_r.inv() * abs_r
                self.episode_buffers['wrist_cam'].append(state_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                self.episode_buffers['side_cam'].append(state_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                self.episode_buffers['tcp_pose'].append(state_data['tcp_pose'])
                self.episode_buffers['joint_pos'].append(state_data['joint_pos'])
                self.episode_buffers['action'].append(np.concatenate((curr_p_action, curr_r_action.as_quat(scalar_first=True), [gripper_action])))
                self.episode_buffers['action_mode'].append(INTV)
                self.j += 1
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
    
    def record_detach_tcp(self):
        """Record current TCP position without informing teleop node"""
        self.detach_tcp = self.robot_env.robot.get_tcp_pose()
        print(f"Detaching sigma at TCP position: {self.detach_tcp}")
    
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
    
    def send_resume_sigma(self):
        """Send resume message to sigma device"""
        msg = f"SIGMA_of_{self.teleop_id}_RESUME_from_{self.robot_id}".encode()
        self.socket.send(msg)
