import pathlib
import sys
import os
import re
import time
import numpy as np
import torch
import dill
import cv2
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer

from robot_env import RobotEnv, HUMAN, ROBOT, PRE_INTV, INTV, postprocess_action_mode
from hardware.my_device.macros import CAM_SERIAL
from util.episode_utils import EpisodeManager


class FailureDetectionModule(ABC):
    """Abstract base class for failure detection modules"""
    
    @abstractmethod
    def initialize(self, cfg: Dict[str, Any], device: torch.device, **kwargs):
        """Initialize the failure detection module"""
        pass
    
    @abstractmethod
    def detect_failure(self, **kwargs) -> Tuple[bool, Optional[str]]:
        """Detect if failure occurred. Returns (failure_flag, failure_reason)"""
        pass
    
    @abstractmethod
    def process_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single step and return any additional data"""
        pass
    
    @abstractmethod
    def finalize_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize episode processing and return any additional data"""
        pass


class RealRobotRunner:
    """Base environment runner for real robot rollouts"""
    
    def __init__(self, eval_cfg, rank: int, device_ids: List[int]):
        self.eval_cfg = eval_cfg
        self.rank = rank
        self.device_ids = device_ids
        self.world_size = len(device_ids)
        self.device_id = device_ids[rank]
        self.device = f"cuda:{self.device_id}"
        self.fps = 10
        
        # Initialize distributed training
        torch.distributed.init_process_group("nccl", rank=rank, world_size=self.world_size)
        
        # Initialize components
        self._load_policy()
        self._setup_transformers()
        self._initialize_robot_env()
        self._initialize_episode_manager()
        self._initialize_replay_buffer()

        self.max_episode_length = self._calculate_max_episode_length()
        
        # Initialize failure detection module if specified
        self.failure_detection_module = None
        if hasattr(eval_cfg, 'failure_detection_module') and eval_cfg.failure_detection_module:
            self._initialize_failure_detection_module()

        # Set random seed
        self.seed = eval_cfg.seed
        np.random.seed(self.seed)
        
        # Set up output directory
        self._setup_output_directory()
        
        # Extract round number for Sirius-specific logic
        self.num_round = self._extract_round_number()
        
        # Episode management
        self.episode_idx = 0 # rollout episode index
        self.episode_id = 0 # replay buffer episode index
        
    def _load_policy(self):
        """Load policy from checkpoint"""
        payload = torch.load(open(self.eval_cfg.checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = payload['cfg']
        
        # Overwrite config values according to evaluation config
        self.cfg.policy.num_inference_steps = self.eval_cfg.policy.num_inference_steps
        self.cfg.output_dir = self.eval_cfg.output_dir
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
        if hasattr(self.eval_cfg, 'Ta'):
            self.Ta = self.eval_cfg.Ta
    
    def _setup_transformers(self):
        """Setup rotation transformers for action and observation spaces"""
        self.action_dim = self.cfg.shape_meta['action']['shape'][0]
        self.action_rot_transformer = None
        self.obs_rot_transformer = None
        
        # Check if there's need for transforming rotation representation
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
        num_samples = getattr(self.eval_cfg, 'num_samples', 1)
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
        zarr_path = os.path.join(self.eval_cfg.train_dataset_path, 'replay_buffer.zarr')
        
        # Determine which keys to load
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=None)

        # Add action_mode if not present
        if 'action_mode' not in self.replay_buffer.keys():
            self.replay_buffer.data['action_mode'] = np.full((self.replay_buffer.n_steps, ), HUMAN)
        
        # Add failure_indices if not present
        if 'failure_indices' not in self.replay_buffer.keys():
            self.replay_buffer.data['failure_indices'] = np.zeros((self.replay_buffer.n_steps, ), dtype=np.bool_)
    
    def _initialize_failure_detection_module(self):
        """Initialize failure detection module based on config"""
        module_name = self.eval_cfg.failure_detection_module
        
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
            cfg=self.eval_cfg,
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
    
    def _setup_output_directory(self):
        """Setup output directory"""
        self.save_img = False
        self.output_dir = os.path.join(self.eval_cfg.output_dir, f"seed_{self.seed}")
        if os.path.isdir(self.output_dir):
            print(f"Output directory {self.output_dir} already exists, will not overwrite it.")
        else:
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
            self.save_img = True
        
        # Create save buffer directory
        os.makedirs(self.eval_cfg.save_buffer_path, exist_ok=True)
    
    def _extract_round_number(self) -> int:
        """Extract round number from save buffer path"""
        match_round = re.search(r'round(\d)', self.eval_cfg.save_buffer_path)
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
            human_eps_len.append(human_episode['side_cam'].shape[0])
        
        return int(torch.max(torch.tensor(human_eps_len)) // self.Ta * self.Ta) if human_eps_len else 600
    
    def run_rollout(self):
        """Main rollout loop"""
        try:
            while True:
                if self.robot_env.keyboard.quit:
                    break
                
                print(f"Rollout episode: {self.episode_idx}")
                if self.eval_cfg.failure_detection_module == 'action_inconsistency_ot':
                    self.failure_detection_module.failure_detector.last_predicted_abs_actions = None
                
                # Run single episode
                episode_data = self._run_single_episode(episode_id=self.episode_id+1)
                
                if episode_data is not None:
                    # Save episode to replay buffer
                    self.replay_buffer.add_episode(episode_data, compressors='disk')
                    self.episode_id = self.replay_buffer.n_episodes - 1
                    print(f'Saved episode {self.episode_id}')
                
                # Reset robot between episodes
                self.robot_env.reset_robot()
                print("Reset!")
                
                self.episode_idx += 1
                
                # Check termination conditions
                if self._should_terminate():
                    break
                
                # For scene configuration reset
                time.sleep(5)
        
        finally:
            self._cleanup()
    
    def _run_single_episode(self, episode_id: int) -> Optional[Dict[str, Any]]:
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
        if getattr(self.eval_cfg, 'random_init', False):
            random_init_pose = self.robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
        
        robot_state = self.robot_env.reset_robot(getattr(self.eval_cfg, 'random_init', False), random_init_pose)
        
        # Initialize episode manager
        self.episode_manager.reset_observation_history()
        
        # Update initial observations
        for _ in range(self.To):
            self.episode_manager.update_observation(
                robot_state['policy_side_img'] / 255.0,
                robot_state['policy_wrist_img'] / 255.0,
                robot_state['tcp_pose'] if self.state_type == 'ee_pose' else robot_state['joint_pos']
            )
        
        # Initialize pose tracking
        if getattr(self.eval_cfg, 'random_init', False) and random_init_pose is not None:
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
            episode_data = self._finalize_episode(episode_buffers, episode_id)
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
                episode_buffers['wrist_cam'].append(state_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                episode_buffers['side_cam'].append(state_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                episode_buffers['tcp_pose'].append(state_data['tcp_pose'])
                episode_buffers['joint_pos'].append(state_data['joint_pos'])
                episode_buffers['action'].append(np.concatenate((curr_p_action, curr_r_action, [gripper_action[0]])))
                episode_buffers['action_mode'].append(ROBOT)
                
                # Update policy observation if needed
                if step >= self.Ta - self.To + 1:
                    self.episode_manager.update_observation(
                        state_data['policy_side_img'] / 255.0,
                        state_data['policy_wrist_img'] / 255.0,
                        state_data['tcp_pose'] if self.state_type == 'ee_pose' else state_data['joint_pos']
                    )
                
                time.sleep(max(1 / self.fps - (time.time() - start_time), 0))
                j += 1
            
            # Store predicted absolute actions for rewinding
            self._last_predicted_abs_actions = predicted_abs_actions
            
            # Process failure detection
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
                        print(f"Failure detected! Due to {failure_reason}")
                    else:
                        print("Maximum episode length reached!")
                    
                    print("Press 'c' to continue; Press 'd' to discard the demo; Press 'h' to request human intervention; Press 'f' to finish the episode.")
                    while not self.robot_env.keyboard.ctn and not self.robot_env.keyboard.discard and not self.robot_env.keyboard.help and not self.robot_env.keyboard.finish:
                        time.sleep(0.1)

                    if self.robot_env.keyboard.finish and failure_flag: # Erase the failure flag because the episode is finished
                        if hasattr(self.failure_detection_module, 'failure_logs'):
                            self.failure_detection_module.failure_logs.popitem()
                        elif hasattr(self.failure_detection_module, 'failure_indices'):
                            self.failure_detection_module.failure_indices.pop()
                    
                    if self.robot_env.keyboard.ctn and j < self.max_episode_length - self.Ta:
                        print("False Positive failure! Continue policy rollout.")
                        self.robot_env.keyboard.ctn = False
                    elif self.robot_env.keyboard.ctn and j >= self.max_episode_length - self.Ta:
                        print("Cannot continue policy rollout, maximum episode length reached. Calling for human intervention.")
                        self.robot_env.keyboard.ctn = False
                        self.robot_env.keyboard.help = True
                        break
            
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
        if self.failure_detection_module:
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
        
        # Reset pose for policy after human intervention
        self.episode_manager.initialize_pose(last_p, last_r.as_quat(scalar_first=True))
        
        # Reset signals
        self.robot_env.keyboard.infer = False
        new_detach_pos, new_detach_rot = self.robot_env.detach_sigma()
        
        return {'timestep': j, 'detach_pos': new_detach_pos, 'detach_rot': new_detach_rot}
    
    def _rewind_robot(self, episode_buffers: Dict[str, List], j: int) -> Tuple[int, np.ndarray, R]:
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
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'should_stop_rewinding') and self.failure_detection_module.enable_OT:
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

                # Rewind the failure logs as well if it's an action inconsistency module
                if self.failure_detection_module and hasattr(self.failure_detection_module, '_rewind_ot_plan'):
                    self.failure_detection_module._rewind_ot_plan(j)
                elif self.failure_detection_module and hasattr(self.failure_detection_module, '_rewind_failure_logs'):
                    self.failure_detection_module._rewind_failure_logs(j)
            
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
    
    def _finalize_episode(self, episode_buffers: Dict[str, List], episode_id: int) -> Dict[str, Any]:
        """Finalize episode and return episode data"""
        episode = dict()
        episode['wrist_cam'] = np.stack(episode_buffers['wrist_cam'], axis=0)
        episode['side_cam'] = np.stack(episode_buffers['side_cam'], axis=0)
        episode['tcp_pose'] = np.stack(episode_buffers['tcp_pose'], axis=0)
        episode['joint_pos'] = np.stack(episode_buffers['joint_pos'], axis=0)
        episode['action'] = np.stack(episode_buffers['action'], axis=0)
        
        # Process action mode
        if getattr(self.eval_cfg, 'post_process_action_mode', False):
            episode['action_mode'] = postprocess_action_mode(np.array(episode_buffers['action_mode']))
        else:
            episode['action_mode'] = np.array(episode_buffers['action_mode'])
        
        assert episode['action_mode'].shape[0] % self.Ta == 0, "A Ta-step chunking is required for the entire demo"
        
        # Finalize failure detection
        if self.failure_detection_module:
            failure_episode_data = self.failure_detection_module.finalize_episode({
                'episode': episode,
                'episode_id': episode_id
            })
            episode.update(failure_episode_data)
        else:
            # Default: no failure indices
            episode['failure_indices'] = np.zeros((episode['action_mode'].shape[0],), dtype=np.bool_)
        
        return episode
    
    def _should_terminate(self) -> bool:
        """Check if rollout should terminate"""
        # Sirius-specific termination condition
        if hasattr(self.eval_cfg, 'sirius_termination') and self.eval_cfg.sirius_termination:
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
        save_zarr_path = os.path.join(self.eval_cfg.save_buffer_path, 'replay_buffer.zarr')
        self.replay_buffer.save_to_path(save_zarr_path)
        
        # Cleanup failure detection module
        if self.failure_detection_module and hasattr(self.failure_detection_module, 'cleanup'):
            self.failure_detection_module.cleanup()
        
        print("Saved replay buffer to", save_zarr_path)
        torch.distributed.destroy_process_group() 