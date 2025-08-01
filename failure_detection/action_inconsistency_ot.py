import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List
from collections import OrderedDict
from torchvision.transforms import Compose, Resize, CenterCrop
import tqdm
import cv2

from diffusion_policy.diffusion_policy.env_runner.real_robot_runner import FailureDetectionModule
from failure_detection.failure_detector import FailureDetector
from robot_env import INTV, HUMAN
from util.image_utils import create_failure_visualization


class ActionInconsistencyOTModule(FailureDetectionModule):
    """Failure detection module using action inconsistency and optimal transport matching"""
    
    def __init__(self):
        self.failure_detector = None
        self.all_human_latent = []
        self.human_demo_indices = []
        self.human_eps_len = []
        self.candidate_expert_indices = []
        self.matched_human_idx = None
        self.human_latent = None
        self.demo_len = None
        self.expert_weight = None
        self.expert_indices = None
        self.greedy_ot_plan = None
        self.greedy_ot_cost = None
        self.rollout_latent = None
        self.action_inconsistency_buffer = []
        self.failure_logs = OrderedDict()
        self.needs_latent = True
        
    def initialize(self, cfg: Dict[str, Any], device: torch.device, **kwargs):
        """Initialize the failure detection module"""
        self.cfg = cfg
        self.device = device
        self.policy = kwargs['policy']
        self.replay_buffer = kwargs['replay_buffer']
        self.episode_manager = kwargs['episode_manager']
        self.Ta = kwargs['Ta']
        self.To = kwargs['To']
        self.ee_pose_dim = kwargs['ee_pose_dim']
        self.img_shape = kwargs['img_shape']
        self.obs_feature_dim = kwargs['obs_feature_dim']
        self.max_episode_length = kwargs['max_episode_length']
        
        # Extract failure detection parameters
        self.enable_action_inconsistency = cfg.enable_action_inconsistency
        self.enable_OT = cfg.enable_OT
        assert self.enable_action_inconsistency or self.enable_OT, "At least one of the evaluation metrics should be enabled"
        self.inconsistency_metric = cfg.inconsistency_metric
        self.num_samples = cfg.num_samples
        self.num_expert_candidates = cfg.num_expert_candidates
        self.action_inconsistency_percentile = cfg.action_inconsistency_percentile
        self.ot_percentile = cfg.ot_percentile
        self.soft_ot_ratio = cfg.soft_ot_ratio
        
        # Initialize failure detector
        self.failure_detector = FailureDetector(
            Ta=self.Ta,
            action_inconsistency_percentile=self.action_inconsistency_percentile,
            ot_percentile=self.ot_percentile,
            max_queue_size=3,
            episode_manager=self.episode_manager,
            enable_action_inconsistency=self.enable_action_inconsistency,
            enable_OT = self.enable_OT
        )
        self.failure_detector.start_async_processing()
        
        # Initialize image processors
        self.side_img_processor = CenterCrop((self.img_shape[1], self.img_shape[2]))
        self.wrist_img_processor = Resize((self.img_shape[1], self.img_shape[2]))
        
        # Prepare human demonstration data
        if self.enable_OT:
            self._prepare_human_demo_data()
        
        # Load previous success statistics if available
        self._load_success_statistics()
        
    def _prepare_human_demo_data(self):
        """Prepare human demonstration data for matching"""
        # Distinguish original human demonstrations from rollouts with human intervention
        self.human_demo_indices = []
        for i in range(self.replay_buffer.n_episodes):
            episode_start = self.replay_buffer.episode_ends[i-1] if i > 0 else 0
            if np.any(self.replay_buffer.data['action_mode'][episode_start: self.replay_buffer.episode_ends[i]] == HUMAN):
                self.human_demo_indices.append(i)
        
        self.all_human_latent = []
        self.human_eps_len = []
        
        for i in tqdm.tqdm(self.human_demo_indices, desc="Obtaining latent for human demo"):
            human_episode = self.replay_buffer.get_episode(i)
            eps_side_img = (self.side_img_processor(torch.from_numpy(human_episode['side_cam']).permute(0, 3, 1, 2)) / 255.0).to(self.device)
            eps_wrist_img = (self.wrist_img_processor(torch.from_numpy(human_episode['wrist_cam']).permute(0, 3, 1, 2)) / 255.0).to(self.device)
            
            # Process state based on type
            if 'tcp_pose' in human_episode:
                eps_state = np.zeros((human_episode['tcp_pose'].shape[0], self.ee_pose_dim))
                eps_state[:, :3] = human_episode['tcp_pose'][:, :3]
                if hasattr(self.episode_manager, 'obs_rot_transformer') and self.episode_manager.obs_rot_transformer:
                    eps_state[:, 3:] = self.episode_manager.obs_rot_transformer.forward(human_episode['tcp_pose'][:, 3:])
                else:
                    eps_state[:, 3:] = human_episode['tcp_pose'][:, 3:]
            else:
                eps_state = human_episode['joint_pos']
            
            eps_state = torch.from_numpy(eps_state).to(self.device)
            demo_len = human_episode['action'].shape[0]
            
            human_latent = torch.zeros((demo_len // self.Ta, int(self.To * self.obs_feature_dim)), device=self.device)
            for idx in range(demo_len // self.Ta):
                human_demo_idx = idx * self.Ta
                if human_demo_idx < self.To - 1:
                    indices = [0] * (self.To - 1 - human_demo_idx) + list(range(human_demo_idx + 1))
                    obs_dict = {
                        'side_img': eps_side_img[indices, :].unsqueeze(0),
                        'wrist_img': eps_wrist_img[indices, :].unsqueeze(0),
                        self.episode_manager.state_type: eps_state[indices, :].unsqueeze(0)
                    }
                else:
                    obs_dict = {
                        'side_img': eps_side_img[human_demo_idx - self.To + 1: human_demo_idx + 1, :].unsqueeze(0),
                        'wrist_img': eps_wrist_img[human_demo_idx - self.To + 1: human_demo_idx + 1, :].unsqueeze(0),
                        self.episode_manager.state_type: eps_state[human_demo_idx - self.To + 1: human_demo_idx + 1, :].unsqueeze(0)
                    }
                
                with torch.no_grad():
                    obs_features = self.policy.extract_latent(obs_dict)
                    human_latent[idx] = obs_features.squeeze(0).reshape(-1)
            
            self.human_eps_len.append(eps_side_img.shape[0])
            self.all_human_latent.append(human_latent)
    
    def _load_success_statistics(self):
        """Load success statistics from previous round if available"""
        import re
        match_round = re.search(r'round(\d)', self.cfg.train_dataset_path)
        training_set_num_round = int(match_round.group(1)) if match_round else 0
        
        match_round = re.search(r'round(\d)', self.cfg.save_buffer_path)
        current_round = int(match_round.group(1)) if match_round else 0
        
        if training_set_num_round != current_round:
            print("Re-initializing success statistics for the current round.")
        else:
            print("Loading success statistics from the previous round.")
            try:
                import os
                prev_success_states = np.load(os.path.join(self.cfg.train_dataset_path, 'success_stats.npz'))
                self.failure_detector.load_success_statistics(prev_success_states)
            except FileNotFoundError:
                print("No previous success statistics found, initializing fresh.")
    
    def detect_failure(self, **kwargs) -> Tuple[bool, Optional[str]]:
        """Detect if failure occurred based on action inconsistency and OT matching"""
        timestep = kwargs.get('timestep', 0)
        max_episode_length = kwargs.get('max_episode_length', self.max_episode_length)
        
        # Get results from the failure detector
        results = self.failure_detector.get_results()
        failure_flag = False
        failure_reason = None
        
        for result in results:
            idx = timestep // self.Ta - 1

            # Process any available async results
            if result["task_type"] == "ot_matching" and result["idx"] <= idx:
                # Update with latest OT results
                self.matched_human_idx = result["matched_human_idx"]
                self.human_latent = result["human_latent"]
                self.demo_len = result["demo_len"]
                self.expert_weight = result["expert_weight"]
                self.expert_indices = result["expert_indices"]
                self.greedy_ot_plan = result["greedy_ot_plan"]
                self.greedy_ot_cost = result["greedy_ot_cost"]
                
                # If action inconsistency is disabled, pad buffer with zeros
                if not self.enable_action_inconsistency:
                    current_buffer_length = len(self.action_inconsistency_buffer)
                    expected_length = (result["idx"] + 1) * self.Ta
                    if current_buffer_length < expected_length:
                        self.action_inconsistency_buffer.extend([0] * (expected_length - current_buffer_length))
                
                # Submit failure detection task
                self.failure_detector.submit_failure_detection_task(
                    action_inconsistency_buffer=self.action_inconsistency_buffer[:int(result["idx"] + 1) * self.Ta].copy(),
                    idx=result["idx"],
                    greedy_ot_cost=self.greedy_ot_cost.clone(),
                    greedy_ot_plan=self.greedy_ot_plan.clone(),
                    max_episode_length=self.max_episode_length
                )
            
            elif result["task_type"] == "action_inconsistency" and result["idx"] <= idx:
                # Track action inconsistency for failure detection
                self.action_inconsistency_buffer.extend([result["action_inconsistency"]] * self.Ta)
                if not self.enable_OT:  # Only when OT is disabled does action inconsistency part submits failure detection task
                    self.failure_detector.submit_failure_detection_task(
                        action_inconsistency_buffer=self.action_inconsistency_buffer[
                                                    :int(result["idx"] + 1) * self.Ta].copy(),
                        idx=result["idx"],
                        greedy_ot_cost=self.greedy_ot_cost.clone(),
                        greedy_ot_plan=self.greedy_ot_plan.clone(),
                        max_episode_length=self.max_episode_length
                    )
            elif result["task_type"] == "failure_detection" and result["idx"] <= idx:
                failure_flag = result["failure_flag"]
                failure_reason = result["failure_reason"]
                
                if failure_flag:
                    if failure_reason == "action inconsistency":
                        failure_type = 1  # ACTION_INCONSISTENCY
                    elif failure_reason == "OT violation":
                        failure_type = 2  # OT
                    
                    self.failure_logs[idx] = failure_type
                    break
        
        # Check for maximum episode length
        if not failure_flag and timestep >= max_episode_length - self.Ta:
            failure_reason = "maximum episode length reached"
        
        return failure_flag, failure_reason
    
    def process_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single step and return any additional data"""
        step_type = step_data['step_type']
        
        if step_type == 'episode_start':
            # Initialize episode-specific data
            self.action_inconsistency_buffer = []
            self.failure_logs = OrderedDict()

            if self.enable_OT:
                # Match the current rollout with the closest expert episode
                rollout_init_latent = self.episode_manager.extract_latent().unsqueeze(0)
                self.candidate_expert_indices = self.episode_manager.find_matching_expert_demo(
                    rollout_init_latent,
                    self.all_human_latent,
                    self.human_demo_indices,
                    self.num_expert_candidates
                )

                self.matched_human_idx = self.human_demo_indices[self.candidate_expert_indices[0]]
                self.human_latent = self.all_human_latent[self.matched_human_idx]
                self.demo_len = self.human_eps_len[self.matched_human_idx]

                # Initialize the rollout optimal transport components
                self.expert_weight = torch.ones((self.demo_len // self.Ta,), device=self.device) / float(
                    self.demo_len // self.Ta)
                self.expert_indices = torch.arange(self.demo_len // self.Ta, device=self.device)
                self.greedy_ot_plan = torch.zeros((self.demo_len // self.Ta, self.max_episode_length // self.Ta),
                                                  device=self.device)
                self.greedy_ot_cost = torch.zeros((self.max_episode_length // self.Ta,), device=self.device)
                self.rollout_latent = torch.zeros(
                    (self.max_episode_length // self.Ta, int(self.To * self.obs_feature_dim)), device=self.device)

                return {'matched_human_idx': self.matched_human_idx}
            else:
                self.greedy_ot_plan = torch.zeros(
                    (self.max_episode_length // self.Ta, self.max_episode_length // self.Ta), device=self.device)
                self.greedy_ot_cost = torch.zeros((self.max_episode_length // self.Ta,), device=self.device)
                return {}  # dummy
            
            return {'matched_human_idx': self.matched_human_idx}
        
        elif step_type == 'policy_step':
            # Process policy step
            action_seq = step_data['action_seq']
            curr_latent = step_data['curr_latent']
            timestep = step_data['timestep']
            predicted_abs_actions = step_data['predicted_abs_actions']
            
            idx = timestep // self.Ta - 1
            
            # Submit action inconsistency task if enabled
            if self.enable_action_inconsistency and idx >= 0:
                self.failure_detector.submit_action_inconsistency_task(
                    action_seq=action_seq,
                    predicted_abs_actions=predicted_abs_actions,
                    idx=idx,
                    last_p=self.episode_manager.last_p,
                    last_r=self.episode_manager.last_r
                )
            
            # Update rollout latent
            if self.enable_OT and idx >= 0 and curr_latent is not None:
                self.rollout_latent[idx] = curr_latent[0].reshape(-1)
                
                # Submit OT matching task
                candidate_expert_latents = [self.all_human_latent[i] for i in self.candidate_expert_indices]
                self.failure_detector.submit_ot_matching_task(
                    rollout_latent=self.rollout_latent.clone(),
                    idx=idx,
                    candidate_expert_latents=candidate_expert_latents,
                    candidate_expert_indices=self.candidate_expert_indices,
                    human_demo_indices=self.human_demo_indices,
                    all_human_latent=self.all_human_latent,
                    human_eps_len=self.human_eps_len,
                    replay_buffer=self.replay_buffer,
                    device=self.device,
                    max_episode_length=self.max_episode_length
                )
        
        return {}
    
    def finalize_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize episode processing and return failure indices"""
        episode = episode_data['episode']
        
        # Wait for final results and empty queues
        self.failure_detector.empty_queue()
        results = self.failure_detector.wait_for_final_results()
        
        for result in results:
            if result["task_type"] == "ot_matching":
                # Update with latest OT results
                self.matched_human_idx = result["matched_human_idx"]
                self.human_latent = result["human_latent"]
                self.demo_len = result["demo_len"]
                self.expert_weight = result["expert_weight"]
                self.expert_indices = result["expert_indices"]
                self.greedy_ot_plan = result["greedy_ot_plan"]
                self.greedy_ot_cost = result["greedy_ot_cost"]
            elif result["task_type"] == "action_inconsistency":
                # Update with latest action inconsistency results
                self.action_inconsistency_buffer.extend([result["action_inconsistency"]] * self.Ta)
        
        self.failure_detector.empty_result_queue()
        
        # Set failure indices in episode data
        failure_signal = np.zeros((episode['action_mode'].shape[0] // self.Ta,), dtype=np.bool_)
        if len(self.failure_logs) > 0:
            failure_signal[list(self.failure_logs.keys())] = 1
        failure_indices = np.repeat(failure_signal, self.Ta)
        
        # Update success statistics
        success = INTV not in episode['action_mode']
        if success:
            success_action_inconsistency = np.array(self.action_inconsistency_buffer).sum()
            self.failure_detector.update_thresholds(
                success_action_inconsistency=success_action_inconsistency,
                greedy_ot_cost=self.greedy_ot_cost,
                timesteps=len(episode['action_mode']) // self.Ta
            )
            
            if len(self.failure_logs) > 0:  # False positive
                ot_fp = 2 in self.failure_logs.values() and self.enable_OT # OT
                action_fp = 1 in self.failure_logs.values() and self.enable_action_inconsistency  # ACTION_INCONSISTENCY
                self.failure_detector.update_percentile_fp(ot_fp=ot_fp, action_fp=action_fp)
                print("False positive trajectory! Raising percentiles...")
            print("OT threshold: ", self.failure_detector.expert_ot_threshold)
            print("Action inconsistency threshold: ", self.failure_detector.expert_action_threshold)
        else:
            # Check for false negative
            has_ot_data = self.enable_OT and len(self.failure_detector.success_ot_values) > 0
            has_action_data = self.enable_action_inconsistency and len(self.failure_detector.success_action_inconsistencies) > 0
            if len(self.failure_logs) == 0 and (has_ot_data or has_action_data):
                self.failure_detector.update_percentile_fn()
                print("False negative trajectory! Lowering percentiles...")
                print("OT threshold: ", self.failure_detector.expert_ot_threshold)
                print("Action inconsistency threshold: ", self.failure_detector.expert_action_threshold)
        
        # Create visualization
        try:
            if self.enable_OT and hasattr(self, 'matched_human_idx') and self.matched_human_idx is not None:
                eps_side_img = (torch.from_numpy(self.replay_buffer.get_episode(self.matched_human_idx)['side_cam']).permute(0, 3, 1, 2) / 255.0).to(self.device)
                action_inconsistency_buffer = np.array(self.action_inconsistency_buffer)
                
                fig = create_failure_visualization(
                    action_inconsistency_buffer=action_inconsistency_buffer,
                    greedy_ot_plan=self.greedy_ot_plan,
                    greedy_ot_cost=self.greedy_ot_cost,
                    human_eps_len=self.human_eps_len[self.matched_human_idx],
                    max_episode_length=self.max_episode_length,
                    Ta=self.Ta,
                    episode=episode,
                    eps_side_img=eps_side_img,
                    demo_len=self.demo_len,
                    failure_indices=list(self.failure_logs.keys())
                )
                
                episode_id = episode_data['episode_id']
                plt.savefig(f'{self.cfg.save_buffer_path}/episode_{episode_id}.png', bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"Failed to create visualization: {e}")
        
        return {'failure_indices': failure_indices}
    
    def should_stop_rewinding(self, j: int, episode_buffers: Dict[str, List], total_greedy_ot_cost: torch.Tensor) -> bool:
        """Check if rewinding should stop based on OT cost and intervention detection"""
        # Adaptively check if OT cost dropped below soft threshold to stop rewinding
        if hasattr(self, 'soft_ot_ratio') and hasattr(self, 'greedy_ot_cost'):
            curr_timestep = j
            soft_ot_threshold = self.soft_ot_ratio * total_greedy_ot_cost
            
            if torch.sum(self.greedy_ot_cost[:j // self.Ta]) < soft_ot_threshold:
                print("OT cost dropped below the soft threshold, stop rewinding.")
                self.failure_detector.last_predicted_abs_actions = None
                return True
        
        # Check if human intervention was detected
        if len(episode_buffers['action_mode']) > 0 and episode_buffers['action_mode'][-1] == INTV:
            print("Human intervention detected, stop rewinding.")
            self.failure_detector.last_predicted_abs_actions = None
            return True
        
        return False
    
    def rewind_step_cleanup(self):
        """Clean up action inconsistency buffer during rewinding"""
        if self.enable_action_inconsistency and len(self.action_inconsistency_buffer) > 0:
            self.action_inconsistency_buffer.pop()
    
    def _rewind_ot_plan(self, j: int):
        """Rewind the OT plan and adjust failure logs"""
        if hasattr(self, 'expert_weight') and hasattr(self, 'greedy_ot_plan'):
            # Rewind the OT plan
            recovered_expert_weight = torch.zeros((self.demo_len // self.Ta,), device=self.device)
            recovered_expert_weight[self.expert_indices] = self.expert_weight.to(recovered_expert_weight.dtype)
            self.expert_weight = recovered_expert_weight + self.greedy_ot_plan[:, j // self.Ta - 1]
            self.expert_indices = torch.nonzero(self.expert_weight)[:, 0]
            self.greedy_ot_plan[:, j // self.Ta - 1] = 0.
            self.greedy_ot_cost[j // self.Ta - 1] = 0.
            
            # Adjust failure indices if needed
            if len(self.failure_logs) > 0:
                latest_failure_timestep, latest_failure_type = self.failure_logs.popitem()
                if latest_failure_timestep == j // self.Ta - 1:
                    for timestep, failure_type in self.failure_logs.items():
                        if timestep >= latest_failure_timestep - 1:
                            self.failure_logs.move_to_end(timestep)
                            _, _ = self.failure_logs.popitem()
                    self.failure_logs[latest_failure_timestep - 1] = latest_failure_type
                else:
                    self.failure_logs[latest_failure_timestep] = latest_failure_type
    
    def cleanup(self):
        """Cleanup resources"""
        if self.failure_detector:
            self.failure_detector.stop_async_processing()
            
            # Save success statistics
            success_stats = self.failure_detector.get_success_statistics()
            import os
            np.savez(os.path.join(self.cfg.save_buffer_path, 'success_stats.npz'), **success_stats)
            print("Saved success statistics") 