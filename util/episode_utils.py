import torch
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from scipy.spatial.transform import Rotation as R

class EpisodeManager:
    def __init__(self, policy, obs_rot_transformer, action_rot_transformer, 
                 obs_feature_dim, img_shape, state_type, state_shape, action_dim,
                 To, Ta, device, num_samples=10):
        """Initialize policy interaction helper
        
        Args:
            policy: The policy model to use for inference
            obs_rot_transformer: Rotation transformer for observations
            action_rot_transformer: Rotation transformer for actions
            obs_feature_dim: Dimensionality of observation features
            img_shape: Shape of input images
            state_type: Type of state ('ee_pose' or 'qpos')
            state_shape: Shape of state
            action_dim: Dimensionality of action
            To: Number of observation steps
            Ta: Number of action steps
            device: Device to run inference on
            num_samples: Number of samples for policy inference
        """
        self.policy = policy
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
        
        # Initialize action tracking
        self.last_predicted_abs_actions = None
        self.curr_predicted_abs_actions = None
        
        # Current pose tracking
        self.last_p = None
        self.last_r = None
        
    def reset_observation_history(self):
        """Reset the observation history buffers"""
        self.policy_img_0_history = torch.zeros((self.To, *self.img_shape), device=self.device)
        self.policy_img_1_history = torch.zeros((self.To, *self.img_shape), device=self.device)
        self.policy_state_history = torch.zeros((self.To, *self.state_shape), device=self.device)
        
    def initialize_pose(self, p, r, num_samples=None):
        """Initialize pose tracking"""
        if num_samples is None:
            num_samples = self.num_samples
            
        self.last_p = p[np.newaxis, :].repeat(num_samples, axis=0)
        self.last_r = R.from_quat(r[np.newaxis, :].repeat(num_samples, axis=0), scalar_first=True)
        
    def preprocess_robot_state(self, state, state_type):
        """Preprocess robot state based on type"""
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
            state = self.preprocess_robot_state(state, self.state_type)
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
    
    def extract_latent(self):
        """Extract latent vector from current observation"""
        obs = {
            'side_img': self.policy_img_0_history.unsqueeze(0),
            'wrist_img': self.policy_img_1_history.unsqueeze(0),
            self.state_type: self.policy_state_history.unsqueeze(0)
        }
        
        with torch.no_grad():
            latent = self.policy.extract_latent(obs)
            return latent.reshape(-1)
    
    def get_absolute_action_for_step(self, action_seq, step):
        """Get absolute action for a specific step in the action chunk"""
        curr_p_action = action_seq[:, step, :3]
        curr_p = self.last_p + curr_p_action
        
        curr_r_action = self.action_rot_transformer.inverse(action_seq[:, step, 3:self.action_dim-1])
        action_rot = R.from_quat(curr_r_action, scalar_first=True)
        curr_r = self.last_r * action_rot
        
        gripper_action = action_seq[:, step, -1]
    
        self.last_p = curr_p
        self.last_r = curr_r

        if step == self.Ta - 1:
            self.last_r = R.from_quat(R.as_quat(self.last_r, scalar_first=True)[0:1, :].repeat(self.num_samples, axis=0), scalar_first=True)
            self.last_p = self.last_p[0:1, :].repeat(self.num_samples, axis=0)
        
        # Return deployed action
        deployed_action = np.concatenate((curr_p[0], curr_r[0].as_quat(scalar_first=True)), 0)
        
        return deployed_action, gripper_action, curr_p, curr_r, curr_p_action[0], curr_r_action[0]
    
    def imagine_future_action(self, action_seq, predicted_abs_actions):
        """Derive the rest of the action chunk and calculate the action inconsistency"""
        tmp_last_p = self.last_p
        tmp_last_r = self.last_r
        for step in range(self.Ta, action_seq.shape[1]):
            curr_p_action = action_seq[:, step, :3]
            curr_p = tmp_last_p + curr_p_action
            curr_r_action = self.action_rot_transformer.inverse(action_seq[:, step, 3:self.action_dim-1])
            action_rot = R.from_quat(curr_r_action, scalar_first=True)
            curr_r = tmp_last_r * action_rot
            predicted_abs_actions[:, step] = np.concatenate((curr_p, curr_r.as_quat(scalar_first=True), action_seq[:, step, -1:]), -1)
            tmp_last_p = curr_p
            tmp_last_r = curr_r
        
        if self.last_predicted_abs_actions is None:
            self.last_predicted_abs_actions = np.concatenate((np.zeros((self.Ta, 8)), predicted_abs_actions[0, :-self.Ta]), 0) # Prevent anomalous value in the beginning
        action_inconsistency = np.mean(np.linalg.norm(predicted_abs_actions[:, :-self.Ta] - self.last_predicted_abs_actions[np.newaxis, self.Ta:], axis=-1))
        self.last_predicted_abs_actions = predicted_abs_actions[0]
        return action_inconsistency
    
    def find_matching_expert_demo(self, rollout_init_latent, all_human_latent, human_demo_indices, num_expert_candidates):
        """Find the most similar expert demonstration"""
        from util.failure_detection_util import cosine_distance, optimal_transport_plan
        
        all_transport_costs = []
        
        for human_latent in all_human_latent:
            cost_mat = cosine_distance(human_latent, rollout_init_latent).to(self.device).detach()
            transport_plan = optimal_transport_plan(human_latent, rollout_init_latent, cost_mat)
            transport_cost = torch.sum(transport_plan * cost_mat, dim=0).squeeze()
            all_transport_costs.append(transport_cost)

        all_transport_costs = torch.stack(all_transport_costs, dim=0)
        _, candidate_expert_indices = torch.topk(all_transport_costs, 
                                                 k=min(len(all_transport_costs), num_expert_candidates), 
                                                 largest=False, 
                                                 sorted=True)
        
        return candidate_expert_indices 