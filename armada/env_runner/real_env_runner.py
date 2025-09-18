import sys
import os
import re
import time
import numpy as np
import torch
import dill
import cv2
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R
from omegaconf import DictConfig
import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from armada.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from armada.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from armada.diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer

from hardware.robot_env import RobotEnv
from hardware.my_device.macros import CAM_SERIAL, HUMAN, ROBOT, INTV
from armada.utils.episode_manager import EpisodeManager
from armada.env_runner.base_env_runner import BaseEnvRunner

class RealEnvRunner(BaseEnvRunner):
    """Environment runner for single-robot case"""
    
    def __init__(self, 
                 cfg: DictConfig, 
                 rank: int, 
                 device_ids: List[int]):
        super().__init__(cfg, rank, device_ids)
        
        # Initialize components
        self._load_policy()
        self._setup_transformers()
        self._initialize_robot_env()
        self._initialize_episode_manager()
        self._initialize_replay_buffer()

        self.max_episode_length = self._calculate_max_episode_length()
        
        # Initialize failure detection module if specified
        self.failure_detection_module = None
        if hasattr(cfg, 'failure_detection'):
            self._initialize_failure_detection_module()

        # Set random seed
        self.seed = cfg.seed
        np.random.seed(self.seed)
        
        # Set up output directory (for scene setup visualization)
        self._setup_output_directory()
        
        # Extract deployment round number, contained in save_buffer_path
        self.num_round = self._extract_round_number()
        
        # Episode number management
        self.episode_idx = 0 # rollout episode index
        self.saved_episode_idx = 0 # replay buffer episode index (only those saved to replay buffer)
        
    def _load_policy(self):
        """Load policy from checkpoint"""
        payload = torch.load(open(self.cfg.checkpoint_path, 'rb'), pickle_module=dill)
        
        # Initialize trained workspace
        cls = hydra.utils.get_class(self.cfg.training._target_)
        workspace = cls(self.cfg.training, self.rank, self.world_size, self.device_id, self.device)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # Get trained policy from workspace
        self.policy = workspace.model.module
        if self.cfg.training.training.use_ema:
            self.policy = workspace.ema_model.module
        
        self.policy.to(self.device)
        self.policy.eval()
        
        # Extract policy parameters
        self.To = self.cfg.training.n_obs_steps
        self.Ta = self.cfg.Ta
        self.obs_feature_dim = self.policy.obs_feature_dim
        self.img_shape = self.cfg.training.task.image_shape
    
    def _setup_transformers(self):
        """Setup rotation transformers for action and observation spaces"""
        self.action_dim = self.cfg.training.shape_meta.action.shape[0]
        self.action_rot_transformer = None
        self.obs_rot_transformer = None
        
        # Check if there's need for transforming rotation representation
        if 'rotation_rep' in self.cfg.training.shape_meta.action:
            self.action_rot_transformer = RotationTransformer(
                from_rep='quaternion', # The robot env uses quaternion representation by default
                to_rep=self.cfg.training.shape_meta.action.rotation_rep
            )
        if 'ee_pose' in self.cfg.training.shape_meta.obs:
            self.ee_pose_dim = self.cfg.training.shape_meta.obs.ee_pose.shape[0]
            self.state_type = 'ee_pose'
            self.state_shape = self.cfg.training.shape_meta.obs.ee_pose.shape
            if 'rotation_rep' in self.cfg.training.shape_meta.obs.ee_pose:
                self.obs_rot_transformer = RotationTransformer(
                    from_rep='quaternion', 
                    to_rep=self.cfg.training.shape_meta.obs.ee_pose.rotation_rep
                )
        else:
            self.ee_pose_dim = self.cfg.training.shape_meta.obs.qpos.shape[0]
            self.state_type = 'qpos'
            self.state_shape = self.cfg.training.shape_meta.obs.qpos.shape
    
    def _initialize_robot_env(self):
        """Initialize robot environment"""
        self.robot_env = RobotEnv(
            camera_serial=CAM_SERIAL, 
            img_shape=self.img_shape, 
            fps=self.fps
        )
    
    def _initialize_episode_manager(self):
        """Initialize episode manager"""
        self.episode_manager = EpisodeManager(
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
            num_samples=self.cfg.failure_detection.num_samples
        )
    
    def _initialize_replay_buffer(self):
        """Initialize replay buffer for data collection"""
        base_zarr_path = os.path.join(self.cfg.train_dataset_path, 'replay_buffer.zarr')
        self.replay_buffer = ReplayBuffer.copy_from_path(base_zarr_path, keys=None) # Build upon previous training set

        # Add action_mode if not present
        if 'action_mode' not in self.replay_buffer.keys():
            self.replay_buffer.data['action_mode'] = np.full((self.replay_buffer.n_steps, ), HUMAN)
        
        # Add failure_indices if not present
        if 'failure_indices' not in self.replay_buffer.keys():
            self.replay_buffer.data['failure_indices'] = np.zeros((self.replay_buffer.n_steps, ), dtype=np.bool_)
    
    def _initialize_failure_detection_module(self):
        """Initialize failure detection module based on config"""
        self.failure_detection_module = hydra.utils.instantiate(self.cfg.failure_detection)
        
        # Initialize the module using runtime variables
        self.failure_detection_module.runtime_initialize(
            device=self.device,
            policy=self.policy,
            replay_buffer=self.replay_buffer,
            episode_manager=self.episode_manager,
            max_episode_length=self.max_episode_length
        )
    
    def _setup_output_directory(self):
        """Setup output directory for scene configuration visualization, usually used for fair comparison between multiple policies"""
        self.save_img = False
        self.output_dir = os.path.join(self.cfg.output_dir, f"seed_{self.seed}")
        if os.path.isdir(self.output_dir): # Precise scene reset for fair policy comparison
            print(f"Output directory {self.output_dir} already exists, will not overwrite it.")
        else: # Start another round of deployment without reference scene setup
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
            self.save_img = True
        
        # Create save buffer directory
        os.makedirs(self.cfg.save_buffer_path, exist_ok=True)
    
    def _extract_round_number(self) -> int:
        """Extract deployment round number from save buffer path """
        match_round = re.search(r'round(\d)', self.cfg.save_buffer_path)
        if match_round:
            return int(match_round.group(1))
        return 0
    
    def _calculate_max_episode_length(self) -> int:
        """Calculate maximum episode length based on expert demonstrations"""
        human_demo_indices = []
        for i in range(self.replay_buffer.n_episodes):
            episode_start = self.replay_buffer.episode_ends[i-1] if i > 0 else 0
            if np.any(self.replay_buffer.data['action_mode'][episode_start: self.replay_buffer.episode_ends[i]] == HUMAN):
                human_demo_indices.append(i)
        
        human_eps_len = []
        for i in human_demo_indices:
            human_episode = self.replay_buffer.get_episode(i)
            human_eps_len.append(human_episode['side_cam'].shape[0])
        
        return int(torch.max(torch.tensor(human_eps_len)) // self.Ta * self.Ta) # Truncate to the nearest multiple of Ta
    
    def run_rollout(self):
        """Main rollout loop"""
        try:
            while True:
                if self.robot_env.keyboard.quit:
                    break
                
                print(f"Rollout episode: {self.episode_idx}")
                
                # Run single episode
                episode_data = self._run_single_episode(saved_episode_idx=self.saved_episode_idx+1)
                
                if episode_data is not None:
                    # Save episode to replay buffer
                    self.replay_buffer.add_episode(episode_data, compressors='disk')
                    self.saved_episode_idx = self.replay_buffer.n_episodes - 1
                    print(f'Saved episode {self.saved_episode_idx}')
                
                # Reset robot between episodes
                self.robot_env.reset_robot()
                print("Reset!")
                
                self.episode_idx += 1
                
                # For scene configuration reset
                time.sleep(5)
        
        finally:
            self._cleanup()
    
    def _run_single_episode(self, saved_episode_idx: int) -> Optional[Dict[str, Any]]:
        """Run a single episode and return episode data"""
        # Reset keyboard states
        self.robot_env.keyboard.finish = False
        self.robot_env.keyboard.help = False
        self.robot_env.keyboard.infer = False
        self.robot_env.keyboard.discard = False
        time.sleep(1)
        
        # Initialize episode buffers
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
        if getattr(self.cfg, 'random_init', False):
            random_init_pose = self.robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
        
        robot_state = self.robot_env.reset_robot(getattr(self.cfg, 'random_init', False), random_init_pose)
        
        # Initialize episode manager
        self.episode_manager.reset_observation_history()
        
        # Update initial observations for policy obs buffer
        for _ in range(self.To):
            self.episode_manager.update_observation(
                robot_state['policy_side_img'] / 255.0,
                robot_state['policy_wrist_img'] / 255.0,
                robot_state['tcp_pose'] if self.state_type == 'ee_pose' else robot_state['joint_pos']
            )
        
        # Initialize target pose tracking
        if getattr(self.cfg, 'random_init', False) and random_init_pose is not None:
            self.episode_manager.initialize_pose(random_init_pose[:3], random_init_pose[3:])
        else:
            self.episode_manager.initialize_pose(self.robot_env.robot.init_pose[:3], self.robot_env.robot.init_pose[3:])
        
        # Scene alignment if there are reference scene setups
        if self.save_img or not os.path.isfile(os.path.join(self.output_dir, f"side_{self.episode_idx}.png")):
            self.robot_env.save_scene_images(self.output_dir, self.episode_idx)
        else:
            self.robot_env.align_scene_with_file(self.output_dir, self.episode_idx)
        
        # Initialize failure detection module step data
        if self.failure_detection_module:
            init_policy_obs = self.episode_manager.get_policy_observation()[0:1] # Keep dim but only require the first sample
            with torch.no_grad():
                init_latent = self.policy.extract_latent(init_policy_obs)
                init_latent = init_latent.reshape(-1)
            self.failure_detection_module.process_step({
                'step_type': 'episode_start',
                'episode_idx': self.episode_idx,
                'rollout_init_latent': init_latent.unsqueeze(0) # For determining expert candidates
            })
        
        # Detach teleop device
        detach_pos, detach_rot = self.robot_env.detach_sigma()
        
        j = 0  # Episode timestep
        
        while True:
            if j >= self.max_episode_length - self.Ta:
                print("Maximum episode length reached, turning to human for help.")
                self.robot_env.keyboard.help = True
            
            # Policy inference loop
            policy_loop_result = self._run_policy_inference_loop(episode_buffers, j)
            j = policy_loop_result['timestep']
            
            if policy_loop_result['break_episode']:
                break
            
            # Human intervention if requested
            if self.robot_env.keyboard.help:
                intervention_result = self._run_human_intervention(episode_buffers, detach_pos, detach_rot, j)
                j = intervention_result['timestep']
                detach_pos, detach_rot = intervention_result['detach_pos'], intervention_result['detach_rot']
            
            # Check if episode should finish
            if self.robot_env.keyboard.discard:
                return None
            
            if self.robot_env.keyboard.finish:
                break
        
        # Finalize episode
        if self.robot_env.keyboard.finish:
            episode_data = self._finalize_episode(episode_buffers)
            return episode_data
        
        return None
    
    def _run_policy_inference_loop(self, episode_buffers: Dict[str, List], j: int) -> Dict[str, Any]:
        """Run policy inference loop"""
        print("=========== Policy inference ============")
        
        while not self.robot_env.keyboard.finish and not self.robot_env.keyboard.discard and not self.robot_env.keyboard.help:
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
                episode_buffers['wrist_cam'].append(state_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                episode_buffers['side_cam'].append(state_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                episode_buffers['tcp_pose'].append(state_data['tcp_pose'])
                episode_buffers['joint_pos'].append(state_data['joint_pos'])
                episode_buffers['action'].append(np.concatenate((curr_p_action, curr_r_action, [gripper_action[0]])))
                episode_buffers['action_mode'].append(ROBOT)
                
                # Update policy observation for the last few steps
                if step >= self.Ta - self.To + 1:
                    self.episode_manager.update_observation(
                        state_data['policy_side_img'] / 255.0,
                        state_data['policy_wrist_img'] / 255.0,
                        state_data['tcp_pose'] if self.state_type == 'ee_pose' else state_data['joint_pos']
                    )
                
                time.sleep(max(1 / self.fps - (time.time() - start_time), 0))
                j += 1
            
            # ================Detect failure===============
            if self.failure_detection_module:
                step_data = {
                    'step_type': 'policy_step',
                    'curr_latent': curr_latent,
                    'timestep': j,
                    'robot_state': robot_state
                }
                
                self.failure_detection_module.process_step(step_data)
                
                failure_flag, failure_reason, _ = self.failure_detection_module.detect_failure(
                    timestep=j,
                    max_episode_length=self.max_episode_length
                )

                if j >= self.max_episode_length - self.Ta: # Making sure that every failure detection result is processed before raising timeout
                    failure_flag, failure_reason, _ = self.failure_detection_module.wait_for_final_results(j)

                print(f"=========== Global timestep: {j // self.Ta - 1} =============")
                
                if failure_flag or j >= self.max_episode_length - self.Ta:
                    if failure_flag:
                        print(f"Failure detected! Due to {failure_reason}")
                    else:
                        print("Maximum episode length reached!")
                    
                    print("Press 'c' to continue; Press 'd' to discard the demo; Press 'h' to request human intervention; Press 'f' to finish the episode.")
                    while not self.robot_env.keyboard.ctn and not self.robot_env.keyboard.discard and not self.robot_env.keyboard.help and not self.robot_env.keyboard.finish:
                        time.sleep(0.1)
                    
                    if self.robot_env.keyboard.ctn and j < self.max_episode_length - self.Ta:
                        print("False Positive failure! Continue policy rollout.")
                        self.robot_env.keyboard.ctn = False
                    elif self.robot_env.keyboard.ctn and j >= self.max_episode_length - self.Ta:
                        print("Cannot continue policy rollout, maximum episode length reached. Calling for human intervention.")
                        self.robot_env.keyboard.ctn = False
                        self.robot_env.keyboard.help = True
                        break
            # ================Detect failure end===============
            
            # Check for maximum episode length without failure detection
            elif j >= self.max_episode_length - self.Ta:
                print("Maximum episode length reached, turning to human for help.")
                self.robot_env.keyboard.help = True
                break
        
        return {'timestep': j, 'break_episode': False}
    
    def _run_human_intervention(self, episode_buffers: Dict[str, List], detach_pos: np.ndarray, detach_rot: R, j: int) -> Dict[str, Any]:
        """Run human intervention loop"""
        print("============ Human intervention =============")
        
        # Reset help signal
        self.robot_env.keyboard.help = False
        
        # Perform rewinding if needed
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'should_stop_rewinding'):
            j, curr_pos, curr_rot = self._rewind_robot(episode_buffers, j)
        
        # Get current pose for human teleop
        last_p = curr_pos if 'curr_pos' in locals() else self.episode_manager.last_p[0]
        last_r = curr_rot if 'curr_rot' in locals() else self.episode_manager.last_r[0]
        
        # Transform sigma device from current robot pose
        translate = last_p - detach_pos
        rotation = detach_rot.inv() * last_r
        self.robot_env.sigma.resume()
        self.robot_env.sigma.transform_from_robot(translate, rotation)
        
        # Human intervention loop
        while (not self.robot_env.keyboard.finish and not self.robot_env.keyboard.discard and not self.robot_env.keyboard.infer) or j % self.Ta:
            # Execute one step of human teleop
            teleop_data, last_p, last_r = self.robot_env.human_teleop_step(last_p, last_r)
            
            if teleop_data is None:
                continue
            
            # Update observation history with latest state
            self.episode_manager.update_observation(
                teleop_data['policy_side_img'] / 255.0,
                teleop_data['policy_wrist_img'] / 255.0,
                teleop_data['tcp_pose'] if self.state_type == 'ee_pose' else teleop_data['joint_pos']
            )
            
            # Store demo data
            episode_buffers['wrist_cam'].append(teleop_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            episode_buffers['side_cam'].append(teleop_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            episode_buffers['tcp_pose'].append(teleop_data['tcp_pose'])
            episode_buffers['joint_pos'].append(teleop_data['joint_pos'])
            episode_buffers['action'].append(teleop_data['action'])
            episode_buffers['action_mode'].append(teleop_data['action_mode'])
            
            j += 1
        
        # Reset target pose tracking after human intervention
        self.episode_manager.initialize_pose(last_p, last_r.as_quat(scalar_first=True))
        
        # Reset signals
        self.robot_env.keyboard.infer = False
        new_detach_pos, new_detach_rot = self.robot_env.detach_sigma()
        
        return {'timestep': j, 'detach_pos': new_detach_pos, 'detach_rot': new_detach_rot}
    
    def _rewind_robot(self, episode_buffers: Dict[str, List], j: int) -> Tuple[int, np.ndarray, R]:
        """Rewind the robot for human intervention"""
        print("Rewinding robot...")
        
        curr_timestep = j
        
        # Use the last target action for rewinding
        curr_pos = self.episode_manager.last_p[0]
        curr_rot = self.episode_manager.last_r[0]
        
        # Let failure detection module determine rewinding behavior
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'should_stop_rewinding'):
            rewind_steps, prev_side_cam, prev_wrist_cam, curr_pos, curr_rot = self._rewind_with_failure_detection(episode_buffers, curr_timestep, curr_pos, curr_rot)
        else:
            rewind_steps, prev_side_cam, prev_wrist_cam, curr_pos, curr_rot = self._rewind_simple(episode_buffers, curr_timestep, curr_pos, curr_rot)
        
        j -= rewind_steps
        
        # Prepare for human intervention by showing reference scene
        if rewind_steps > 0:
            print("Please reset the scene and press 'c' to go on to human intervention")
            ref_side_img = cv2.cvtColor(prev_side_cam, cv2.COLOR_RGB2BGR)
            ref_wrist_img = cv2.cvtColor(prev_wrist_cam, cv2.COLOR_RGB2BGR)
            self.robot_env.align_with_reference(ref_side_img, ref_wrist_img)
        
        return j, curr_pos, curr_rot
    
    def _rewind_with_failure_detection(self, episode_buffers: Dict[str, List], curr_timestep: int, curr_pos: np.ndarray, curr_rot: R) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, R]:
        """Rewind with failure detection module guidance"""
        rewind_steps = 0
        j = curr_timestep
        
        for _ in range(curr_timestep):
            if not self.failure_detection_module.rewind_step(j, episode_buffers, curr_timestep):
                break
            # Rewind one step on robot
            curr_pos, curr_rot, prev_side_cam, prev_wrist_cam = self._rewind_single_step(episode_buffers, curr_pos, curr_rot)
            j -= 1
            rewind_steps += 1
        
        return rewind_steps, prev_side_cam, prev_wrist_cam, curr_pos, curr_rot
    
    def _rewind_simple(self, episode_buffers: Dict[str, List], curr_timestep: int, curr_pos: np.ndarray, curr_rot: R) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, R]:
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
    
    def _rewind_single_step(self, episode_buffers: Dict[str, List], curr_pos: np.ndarray, curr_rot: R) -> Tuple[np.ndarray, R, np.ndarray, np.ndarray]:
        """Rewind a single step"""
        # Get previous action data to rewind
        prev_wrist_cam = episode_buffers['wrist_cam'].pop()
        prev_side_cam = episode_buffers['side_cam'].pop()
        episode_buffers['tcp_pose'].pop()
        episode_buffers['joint_pos'].pop()
        prev_action = episode_buffers['action'].pop()
        episode_buffers['action_mode'].pop()
        
        # Rewind robot by applying inverse action
        curr_pos, curr_rot = self.robot_env.rewind_robot(curr_pos, curr_rot, prev_action)
        
        return curr_pos, curr_rot, prev_side_cam, prev_wrist_cam
    
    def _finalize_episode(self, episode_buffers: Dict[str, List]) -> Dict[str, Any]:
        """Finalize episode and return episode data"""
        episode = dict()
        episode['wrist_cam'] = np.stack(episode_buffers['wrist_cam'], axis=0)
        episode['side_cam'] = np.stack(episode_buffers['side_cam'], axis=0)
        episode['tcp_pose'] = np.stack(episode_buffers['tcp_pose'], axis=0)
        episode['joint_pos'] = np.stack(episode_buffers['joint_pos'], axis=0)
        episode['action'] = np.stack(episode_buffers['action'], axis=0)
        episode['action_mode'] = np.array(episode_buffers['action_mode'])
        
        assert episode['action_mode'].shape[0] % self.Ta == 0, "A Ta-step chunking is required for the entire demo"
        
        # Finalize failure detection
        if self.failure_detection_module:
            failure_episode_data = self.failure_detection_module.finalize_episode(episode)
            episode.update(failure_episode_data)
        else:
            # Default: no failure indices
            episode['failure_indices'] = np.zeros((episode['action_mode'].shape[0],), dtype=np.bool_)
        
        return episode
    
    def _cleanup(self):
        """Cleanup resources"""
        # Save the replay buffer
        save_zarr_path = os.path.join(self.cfg.save_buffer_path, 'replay_buffer.zarr')
        self.replay_buffer.save_to_path(save_zarr_path)
        
        # Cleanup failure detection module
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'cleanup'):
            self.failure_detection_module.cleanup()
        
        print("Saved replay buffer to", save_zarr_path)
        torch.distributed.destroy_process_group() 