import numpy as np
import torch
import argparse
import dill
import hydra
import tqdm

from util.ot_util import *
from third_party.diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from third_party.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from third_party.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply


HUMAN = 0
ROBOT = 1
PRE_INTV = 2
INTV = 3

def main(args):
    replay_buffer = ReplayBuffer.copy_from_path(args.dataset_path, keys=['wrist_cam', 'side_cam', \
        'joint_pos', 'action', 'tcp_pose', 'action_mode'])
    
    # Hack environment variables
    world_size = 1
    device_id = 0
    rank = 0
    device = f"cuda:{device_id}"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Load the current checkpoint
    payload = torch.load(open(args.checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, rank, world_size, device_id, device)
    # workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model.module
    if cfg.training.use_ema:
        policy = workspace.ema_model.module
    policy.to(device)
    policy.eval()
    
    # Load some key attributes from the config
    To = cfg.n_obs_steps
    obs_feature_dim = policy.obs_feature_dim
    
    # Distinguish original human demonstrations from rollouts with human intervention
    human_demo_indices = []
    rollout_indices = []
    for i in range(replay_buffer.n_episodes):
        episode_start = replay_buffer.episode_ends[i-1] if i > 0 else 0
        if np.any(replay_buffer.data['action_mode'][episode_start: replay_buffer.episode_ends[i]] == HUMAN):
            human_demo_indices.append(i)
        elif np.any(replay_buffer.data['action_mode'][episode_start: replay_buffer.episode_ends[i]] == INTV):
            rollout_indices.append(i)
            
    for i in tqdm.tqdm(human_demo_indices, desc="Obtaining latent for human demo"):
        rollout_episode = replay_buffer.get_episode(i)
        eps_side_img = (torch.from_numpy(rollout_episode['side_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_wrist_img = (torch.from_numpy(rollout_episode['wrist_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_state = (torch.from_numpy(rollout_episode['tcp_pose'])).to(device)
        rollout_len = rollout_episode['action'].shape[0]
        human_latent = torch.zeros((rollout_len, int(To*obs_feature_dim)), device=device)
        
        for idx in range(rollout_len):
            if idx < To - 1:
                indices = [0] * (To-1-idx) + list(range(idx+1))
                obs_dict = {
                    'side_img': eps_side_img[indices, :].unsqueeze(0),
                    'wrist_img': eps_wrist_img[indices, :].unsqueeze(0), 
                    'ee_pose': eps_state[indices, :].unsqueeze(0)
                }
            else:
                obs_dict = {
                    'side_img': eps_side_img[idx-To+1: idx+1, :].unsqueeze(0),
                    'wrist_img': eps_wrist_img[idx-To+1: idx+1, :].unsqueeze(0), 
                    'ee_pose': eps_state[idx-To+1: idx+1, :].unsqueeze(0)
                }

            obs_features = policy.extract_latent(obs_dict)
            human_latent[idx] = obs_features.squeeze(0).reshape(-1)
            
    for j in tqdm.tqdm(rollout_indices, desc="Obtaining latent for rollouts"):
        rollout_episode = replay_buffer.get_episode(j)
        eps_side_img = (torch.from_numpy(rollout_episode['side_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_wrist_img = (torch.from_numpy(rollout_episode['wrist_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_state = (torch.from_numpy(rollout_episode['tcp_pose'])).to(device)
        rollout_len = rollout_episode['action'].shape[0]
        rollout_latent = torch.zeros((rollout_len, int(To*obs_feature_dim)), device=device)
        
        for idx in range(rollout_len):
            if idx < To - 1:
                indices = [0] * (To-1-idx) + list(range(idx+1))
                obs_dict = {
                    'side_img': eps_side_img[indices, :].unsqueeze(0),
                    'wrist_img': eps_wrist_img[indices, :].unsqueeze(0), 
                    'ee_pose': eps_state[indices, :].unsqueeze(0)
                }
            else:
                obs_dict = {
                    'side_img': eps_side_img[idx-To+1: idx+1, :].unsqueeze(0),
                    'wrist_img': eps_wrist_img[idx-To+1: idx+1, :].unsqueeze(0), 
                    'ee_pose': eps_state[idx-To+1: idx+1, :].unsqueeze(0)
                }

            obs_features = policy.extract_latent(obs_dict)
            rollout_latent[idx] = obs_features.squeeze(0).reshape(-1)
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-temp', '--temperature', type=float, default=1.0)
    parser.add_argument('-p', '--dataset_path', type=str, required=True)
    parser.add_argument('-ckpt', '--checkpoint_path', type=str, required=True)
    args = parser.parse_args()
    
    main(args)