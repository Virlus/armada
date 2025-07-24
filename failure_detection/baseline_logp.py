import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../FAIL_DETECT'))

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

from diffusion_policy.diffusion_policy.env_runner.real_robot_runner import FailureDetectionModule
from diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply


class BaselineLogpModule(FailureDetectionModule):
    """Failure detection module using baseline logpZO_UQ approach"""
    
    def __init__(self):
        self.baseline_model = None
        self.baseline_normalizer = None
        self.logpZO_upper_bound = None
        self.failure_indices = []
        self.needs_latent = False
        
    def initialize(self, cfg: Dict[str, Any], device: torch.device, **kwargs):
        """Initialize the baseline failure detection module"""
        self.cfg = cfg
        self.device = device
        self.policy = kwargs['policy']
        self.Ta = kwargs['Ta']
        self.To = kwargs['To']
        
        # Load baseline model and normalizer
        self.baseline_model = self._get_baseline_model(cfg.baseline_model_path, device)
        self.baseline_normalizer = torch.load(cfg.baseline_normalizer_path)
        self.baseline_normalizer.to(device)
        
        # Load baseline statistics
        self.logpZO_upper_bound = np.load(cfg.baseline_stats_path)['target_traj']
        
        print(f"Loaded baseline model from {cfg.baseline_model_path}")
        print(f"Loaded baseline normalizer from {cfg.baseline_normalizer_path}")
        print(f"Loaded baseline statistics from {cfg.baseline_stats_path}")
        
    def _get_baseline_model(self, model_path: str, device: torch.device):
        """Load baseline model from path"""
        # Import here to avoid circular imports
        from FAIL_DETECT.eval import get_baseline_model
        return get_baseline_model(model_path, device=device)
    
    def detect_failure(self, **kwargs) -> Tuple[bool, Optional[str]]:
        """Detect if failure occurred using baseline logpZO_UQ"""
        timestep = kwargs.get('timestep', 0)
        failure_step_data = kwargs.get('failure_step_data', {})
        max_episode_length = kwargs.get('max_episode_length', 600)
        
        # Get current logp from failure data
        curr_logp = failure_step_data.get('curr_logp', 0.0)
        step_idx = timestep // self.Ta - 1
        
        # Check if failure occurred
        failure_flag = False
        failure_reason = None
        
        if step_idx >= 0 and len(self.logpZO_upper_bound) > step_idx:
            if curr_logp > self.logpZO_upper_bound[step_idx]:
                failure_flag = True
                failure_reason = "baseline logp threshold exceeded"
                self.failure_indices.append(step_idx)
        
        # Check for maximum episode length
        if timestep >= max_episode_length - self.Ta:
            failure_flag = False
            failure_reason = "maximum episode length reached"
        
        return failure_flag, failure_reason
    
    def process_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single step and return logp data"""
        step_type = step_data['step_type']
        
        if step_type == 'episode_start':
            # Initialize episode-specific data
            self.failure_indices = []
            return {}
        
        elif step_type == 'policy_step':
            # Process policy step for baseline failure detection
            policy_obs = step_data['policy_obs']
            timestep = step_data['timestep']
            
            # Calculate baseline logp
            from FAIL_DETECT.eval import logpZO_UQ
            
            normalized_obs = self.baseline_normalizer.normalize(policy_obs)
            this_nobs = dict_apply(normalized_obs, lambda x: x[:, :self.policy.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.policy.obs_encoder.get_dense_feats(this_nobs)
            global_cond = nobs_features.reshape(1, -1)
            curr_logp = logpZO_UQ(self.baseline_model, global_cond, None)
            
            step_idx = timestep // self.Ta - 1
            upper_bound = self.logpZO_upper_bound[step_idx] if step_idx < len(self.logpZO_upper_bound) else float('inf')
            
            print(f"Step {step_idx}: logp={curr_logp.item():.4f}, threshold={upper_bound:.4f}")
            
            return {'curr_logp': curr_logp.item(), 'upper_bound': upper_bound}
        
        return {}
    
    def finalize_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize episode processing and return failure indices"""
        episode = episode_data['episode']
        
        # Set failure indices in episode data
        failure_signal = np.zeros((episode['action_mode'].shape[0] // self.Ta,), dtype=np.bool_)
        if len(self.failure_indices) > 0:
            failure_signal[self.failure_indices] = 1
        failure_indices = np.repeat(failure_signal, self.Ta)
        
        return {'failure_indices': failure_indices} 