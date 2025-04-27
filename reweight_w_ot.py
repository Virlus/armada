import numpy as np
import torch
import argparse
import dill
import hydra
import tqdm
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import BlendedGenericTransform

sys.path.append(os.path.join(os.path.dirname((__file__)), 'third_party/diffusion_policy'))

from util.ot_util import *
from third_party.diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from third_party.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from third_party.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from third_party.diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer


HUMAN = 0
ROBOT = 1
PRE_INTV = 2
INTV = 3


def main(args):
    replay_buffer = ReplayBuffer.copy_from_path(args.dataset_path, keys=['wrist_cam', 'side_cam', \
        'joint_pos', 'action', 'tcp_pose', 'action_mode'])
    save_buffer = ReplayBuffer.create_empty_numpy()
    
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
    shape_meta = cfg.shape_meta
    obs_feature_dim = policy.obs_feature_dim
    n_skip_frame = args.skip_frame
    
    if 'ee_pose' in shape_meta['obs']:
        ee_pose_dim = shape_meta['obs']['ee_pose']['shape'][0]
        if 'rotation_rep' in shape_meta['obs']['ee_pose']:
            obs_rot_transformer = RotationTransformer(from_rep='quaternion', to_rep=shape_meta['obs']['ee_pose']['rotation_rep'])
    
    # Distinguish original human demonstrations from rollouts with human intervention
    human_demo_indices = []
    rollout_indices = []
    for i in range(replay_buffer.n_episodes):
        episode_start = replay_buffer.episode_ends[i-1] if i > 0 else 0
        if np.any(replay_buffer.data['action_mode'][episode_start: replay_buffer.episode_ends[i]] == HUMAN):
            human_demo_indices.append(i)
        elif np.any(replay_buffer.data['action_mode'][episode_start: replay_buffer.episode_ends[i]] == INTV):
            rollout_indices.append(i)
            
    human_init_latent = torch.zeros((len(human_demo_indices), int(To*obs_feature_dim)), device=device)
    rollout_init_latent = torch.zeros((len(rollout_indices), int(To*obs_feature_dim)), device=device)
            
    for i in tqdm.tqdm(human_demo_indices, desc="Obtaining latent for human demo"):
        human_episode = replay_buffer.get_episode(i)
        eps_side_img = (torch.from_numpy(human_episode['side_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_wrist_img = (torch.from_numpy(human_episode['wrist_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_state = np.zeros((human_episode['tcp_pose'].shape[0], ee_pose_dim))
        eps_state[:, :3] = human_episode['tcp_pose'][:, :3]
        eps_state[:, 3:] = obs_rot_transformer.forward(human_episode['tcp_pose'][:, 3:])
        eps_state = (torch.from_numpy(eps_state)).to(device)
        
        indices = [0] * To
        obs_dict = {
            'side_img': eps_side_img[indices, :].unsqueeze(0),
            'wrist_img': eps_wrist_img[indices, :].unsqueeze(0), 
            'ee_pose': eps_state[indices, :].unsqueeze(0)
        }
        obs_features = policy.extract_latent(obs_dict)
        human_init_latent[i] = obs_features.squeeze(0).reshape(-1)
            
    for j, rollout_idx in enumerate(tqdm.tqdm(rollout_indices, desc="Obtaining latent for rollouts")):
        rollout_episode = replay_buffer.get_episode(rollout_idx)
        eps_side_img = (torch.from_numpy(rollout_episode['side_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_wrist_img = (torch.from_numpy(rollout_episode['wrist_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_state = np.zeros((rollout_episode['tcp_pose'].shape[0], ee_pose_dim))
        eps_state[:, :3] = rollout_episode['tcp_pose'][:, :3]
        eps_state[:, 3:] = obs_rot_transformer.forward(rollout_episode['tcp_pose'][:, 3:])
        eps_state = (torch.from_numpy(eps_state)).to(device)
        
        indices = [0] * To
        obs_dict = {
            'side_img': eps_side_img[indices, :].unsqueeze(0),
            'wrist_img': eps_wrist_img[indices, :].unsqueeze(0), 
            'ee_pose': eps_state[indices, :].unsqueeze(0)
        }

        obs_features = policy.extract_latent(obs_dict)
        rollout_init_latent[j] = obs_features.squeeze(0).reshape(-1)
    
    dist_mat = euclidean_distance(human_init_latent, rollout_init_latent)
    dist_mat = dist_mat.to(device).detach()
    human_corr_indices = torch.argmin(dist_mat, dim=0)
    
    # Carry out optimal transport between corresponding episodes
    for k, human_corr_idx in enumerate(tqdm.tqdm(human_corr_indices, desc="OT matching between corresponding episodes")):
        human_episode = replay_buffer.get_episode(human_corr_idx)
        rollout_episode = replay_buffer.get_episode(rollout_indices[k])
        
        eps_side_img = (torch.from_numpy(human_episode['side_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_wrist_img = (torch.from_numpy(human_episode['wrist_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_state = np.zeros((human_episode['tcp_pose'].shape[0], ee_pose_dim))
        eps_state[:, :3] = human_episode['tcp_pose'][:, :3]
        eps_state[:, 3:] = obs_rot_transformer.forward(human_episode['tcp_pose'][:, 3:])
        eps_state = (torch.from_numpy(eps_state)).to(device)
        rollout_len = human_episode['action'].shape[0]
        human_latent = torch.zeros((rollout_len // n_skip_frame, int(To*obs_feature_dim)), device=device)
        
        for idx in range(rollout_len // n_skip_frame):
            episode_idx = idx * n_skip_frame
            if episode_idx < To - 1:
                indices = [0] * (To-1-episode_idx) + list(range(episode_idx+1))
                obs_dict = {
                    'side_img': eps_side_img[indices, :].unsqueeze(0),
                    'wrist_img': eps_wrist_img[indices, :].unsqueeze(0), 
                    'ee_pose': eps_state[indices, :].unsqueeze(0)
                }
            else:
                obs_dict = {
                    'side_img': eps_side_img[episode_idx-To+1: episode_idx+1, :].unsqueeze(0),
                    'wrist_img': eps_wrist_img[episode_idx-To+1: episode_idx+1, :].unsqueeze(0), 
                    'ee_pose': eps_state[episode_idx-To+1: episode_idx+1, :].unsqueeze(0)
                }

            obs_features = policy.extract_latent(obs_dict)
            human_latent[idx] = obs_features.squeeze(0).reshape(-1)
            
        eps_side_img = (torch.from_numpy(rollout_episode['side_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_wrist_img = (torch.from_numpy(rollout_episode['wrist_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_state = np.zeros((rollout_episode['tcp_pose'].shape[0], ee_pose_dim))
        eps_state[:, :3] = rollout_episode['tcp_pose'][:, :3]
        eps_state[:, 3:] = obs_rot_transformer.forward(rollout_episode['tcp_pose'][:, 3:])
        eps_state = (torch.from_numpy(eps_state)).to(device)
        rollout_len = rollout_episode['action'].shape[0]
        rollout_latent = torch.zeros((rollout_len // n_skip_frame, int(To*obs_feature_dim)), device=device)
        
        for idx in range(rollout_len // n_skip_frame):
            episode_idx = idx * n_skip_frame
            if idx < To - 1:
                indices = [0] * (To-1-episode_idx) + list(range(episode_idx+1))
                obs_dict = {
                    'side_img': eps_side_img[indices, :].unsqueeze(0),
                    'wrist_img': eps_wrist_img[indices, :].unsqueeze(0), 
                    'ee_pose': eps_state[indices, :].unsqueeze(0)
                }
            else:
                obs_dict = {
                    'side_img': eps_side_img[episode_idx-To+1: episode_idx+1, :].unsqueeze(0),
                    'wrist_img': eps_wrist_img[episode_idx-To+1: episode_idx+1, :].unsqueeze(0), 
                    'ee_pose': eps_state[episode_idx-To+1: episode_idx+1, :].unsqueeze(0)
                }

            obs_features = policy.extract_latent(obs_dict)
            rollout_latent[idx] = obs_features.squeeze(0).reshape(-1)
            
        dist_mat = euclidean_distance(human_latent, rollout_latent)
        dist_mat = dist_mat.to(device).detach()
        ot_res = optimal_transport_plan(human_latent, rollout_latent, dist_mat)
        ot_cost = torch.sum(ot_res * dist_mat, dim=0)
        
        # Compute the target weight (need to think over this)
        target_weight = torch.zeros((rollout_latent.shape[0],), device=device) # Assume this is derived already
        original_rollout_eps = replay_buffer.get_episode(rollout_indices[k])
        original_rollout_eps['action_weight'] = target_weight.detach().cpu().numpy()
        save_buffer.add_episode(original_rollout_eps)
        
    save_buffer.save_to_path(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-temp', '--temperature', type=float, default=1.0)
    parser.add_argument('-p', '--dataset_path', type=str, required=True)
    parser.add_argument('-save', '--save_path', type=str, required=True)
    parser.add_argument('-ckpt', '--checkpoint_path', type=str, required=True)
    parser.add_argument('-skip', '--skip_frame', type=int, required=True)
    args = parser.parse_args()
    
    os.environ["MASTER_ADDR"] = "localhost"
    port = 29999
    import socket
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                print(port)
                break
        except OSError:
            port += 1
    os.environ["MASTER_PORT"] = f"{port}"
    
    main(args)