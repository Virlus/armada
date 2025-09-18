import torch
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from scipy.spatial.transform import Rotation as R

class EpisodeManager:
    def __init__(self, 
                 obs_rot_transformer=None, 
                 action_rot_transformer=None, 
                 obs_feature_dim: int = None, 
                 img_shape: Tuple[int, int, int] = None, 
                 state_type: str = None, 
                 state_shape: Tuple[int, ...] = None, 
                 action_dim: int = None,
                 To: int = None, 
                 Ta: int = None, 
                 device: torch.device = None, 
                 num_samples: int = None):
        """
        Initialize episode manager
        """
        self.obs_rot_transformer = obs_rot_transformer
        self.action_rot_transformer = action_rot_transformer
        self.obs_feature_dim = obs_feature_dim
        self.img_shape = img_shape
        self.state_type = state_type
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.To = To
        self.Ta = Ta
        self.device = device
        self.num_samples = num_samples
        
        # Initialize observation history buffers
        self.reset_observation_history()
        
        # Target pose tracking
        self.last_p = None
        self.last_r = None
        
    def reset_observation_history(self):
        """Reset the observation history buffers"""
        self.policy_img_0_history = torch.zeros((self.To, *self.img_shape), device=self.device)
        self.policy_img_1_history = torch.zeros((self.To, *self.img_shape), device=self.device)
        self.policy_state_history = torch.zeros((self.To, *self.state_shape), device=self.device)
        
    def initialize_pose(self, p, r):
        """Initialize target pose tracking"""
            
        self.last_p = p[np.newaxis, :].repeat(self.num_samples, axis=0)
        self.last_r = R.from_quat(r[np.newaxis, :].repeat(self.num_samples, axis=0), scalar_first=True)
        
    def _preprocess_robot_state(self, state):
        """Preprocess robot state based on proprioceptive type"""
        state = np.array(state)
        
        if self.obs_rot_transformer is not None:
            tmp_state = np.zeros((self.state_shape[0],))
            tmp_state[:3] = state[:3]
            tmp_state[3:] = self.obs_rot_transformer.forward(state[3:])
            state = tmp_state
            
        return torch.from_numpy(state).to(self.device)
    
    def update_observation(self, side_img, wrist_img, state):
        """Update observation history with new images and state"""
        # Process state if needed
        if not isinstance(state, torch.Tensor):
            state = self._preprocess_robot_state(state)
        # Update observation history
        self.policy_img_0_history = torch.cat([self.policy_img_0_history[1:], side_img.to(self.device).unsqueeze(0)], dim=0)
        self.policy_img_1_history = torch.cat([self.policy_img_1_history[1:], wrist_img.to(self.device).unsqueeze(0)], dim=0)
        self.policy_state_history = torch.cat([self.policy_state_history[1:], state.unsqueeze(0)], dim=0)
    
    def get_policy_observation(self):
        """Get current policy observation dict"""
        return {
            'side_img': self.policy_img_0_history.unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1),
            'wrist_img': self.policy_img_1_history.unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1),
            self.state_type: self.policy_state_history.unsqueeze(0).repeat(self.num_samples, 1, 1)
        }
    
    def get_absolute_action_for_step(self, action_seq, step):
        """Get absolute action for a specific step in the action chunk, used for deployment"""
        curr_p_action = action_seq[:, step, :3]
        curr_p = self.last_p + curr_p_action
        
        curr_r_action = self.action_rot_transformer.inverse(action_seq[:, step, 3:self.action_dim-1])
        action_rot = R.from_quat(curr_r_action, scalar_first=True)
        curr_r = self.last_r * action_rot
        
        gripper_action = action_seq[:, step, -1]
    
        self.last_p = curr_p
        self.last_r = curr_r

        if step == self.Ta - 1: # At the end of the chunk, the target pose of the first sample should override all the others
                                # because the first sample is the one that is actually executed
            self.last_r = R.from_quat(R.as_quat(self.last_r, scalar_first=True)[0:1, :].repeat(self.num_samples, axis=0), scalar_first=True)
            self.last_p = self.last_p[0:1, :].repeat(self.num_samples, axis=0)
        
        # Return deployed action
        deployed_action = np.concatenate((curr_p[0], curr_r[0].as_quat(scalar_first=True)), 0)
        
        return deployed_action, gripper_action, curr_p, curr_r, curr_p_action[0], curr_r_action[0]
 