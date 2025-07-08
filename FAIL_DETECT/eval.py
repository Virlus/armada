import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../diffusion_policy"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import zarr
import numpy as np
import hydra
import pathlib
from omegaconf import OmegaConf
from torch.multiprocessing import Process

from data_loader import adjust_xshape
from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply, dict_apply_with_key
from timeseries_cp.utils.data_utils import RegressionType
from timeseries_cp.methods.functional_predictor import FunctionalPredictor, ModulationType
import CFM.net_CFM as Net


def get_baseline_model(ckpt_file, input_dim=10, device='cuda:0'):
    net = Net.get_unet(input_dim).to(device)
    ckpt = torch.load(ckpt_file)
    net.load_state_dict(ckpt['model'], strict=True)
    return net.eval()


def logpZO_UQ(baseline_model, observation, action_pred = None, input_dim=10):
    observation = observation
    observation = adjust_xshape(observation, input_dim)
    if action_pred is not None:
        action_pred = action_pred
        observation = torch.cat([observation, action_pred], dim=1)
    with torch.no_grad():
        timesteps = torch.zeros(observation.shape[0], device=observation.device)
        pred_v = baseline_model(observation, timesteps)
        observation = observation + pred_v
        logpZO = observation.reshape(len(observation), -1).pow(2).sum(dim=-1)
    return logpZO


def get_episode_length(replay_buffer, Ta=8):
    episode_lengths = []
    for episode_id in range(replay_buffer.n_episodes):
        episode_length = replay_buffer.episode_ends[episode_id] - replay_buffer.episode_ends[episode_id-1] \
            if episode_id > 0 else replay_buffer.episode_ends[episode_id]
        assert episode_length % Ta == 0
        episode_lengths.append(episode_length // Ta)
    return np.max(np.array(episode_lengths))


def get_episode_and_return_logpZO(dataset, policy, baseline_model, replay_buffer, episode_id, episode_length, input_dim=10, \
                                  To=2, Ta=8, device='cuda:0'):
    episode = replay_buffer.get_episode(episode_id)
    curr_episode_length = replay_buffer.episode_ends[episode_id] - replay_buffer.episode_ends[episode_id-1] \
        if episode_id > 0 else replay_buffer.episode_ends[episode_id]
    curr_logpZO = []
    for i in range(episode_length):
        if i * Ta < To - 1:
            obs_indices = [0] * (To - 1 - i * Ta) + list(range(i * Ta + 1))
        elif i * Ta >= curr_episode_length:
            j = curr_episode_length // Ta - 1
            obs_indices = list(range(j * Ta - To + 1, j * Ta + 1))
        else:
            obs_indices = list(range(i * Ta - To + 1, i * Ta + 1))
        wrist_img = np.moveaxis(episode['wrist_cam'][obs_indices], -1, 1) / 255
        side_img = np.moveaxis(episode['side_cam'][obs_indices], -1, 1) / 255
        tcp_pose = episode['tcp_pose'][obs_indices]
        if 'ee_pose' in dataset.shape_meta['obs']:
            ee_pose = np.zeros((tcp_pose.shape[0], dataset.ee_pose_dim))
            ee_pose[:, :3] = tcp_pose[:, :3]
            rel_ee_rot = tcp_pose[:, 3:7]
            if dataset.obs_rot_transformer is not None:
                ee_pose[:, 3:dataset.ee_pose_dim] = dataset.obs_rot_transformer.forward(rel_ee_rot)
            else:
                ee_pose[:, 3:7] = rel_ee_rot

        data = {
            'obs': {
                'wrist_img': wrist_img,
                'side_img': side_img,
                'ee_pose': ee_pose,
            }
        }
        torch_data = dict_apply(data, torch.from_numpy)
        torch_data = dict_apply_with_key(torch_data, dataset.side_image_postprocess, ['side_img'])
        torch_data = dict_apply_with_key(torch_data, dataset.wrist_image_postprocess, ['wrist_img'])
        torch_data = dict_apply(torch_data, lambda x: x.unsqueeze(0))
        batch = dict_apply(torch_data, lambda x: x.to(device, non_blocking=True))
        normalized_data = policy.normalizer.normalize(batch['obs'])
        this_nobs = dict_apply(normalized_data, lambda x: x[:,:policy.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = policy.obs_encoder.get_dense_feats(this_nobs)
        global_cond = nobs_features.reshape(1, -1)
        curr_logpZO.append(logpZO_UQ(baseline_model, global_cond, None, input_dim))
    
    curr_logpZO = torch.stack(curr_logpZO, dim=0)
    assert curr_logpZO.shape[0] == episode_length
    return curr_logpZO


def main(rank, cfg, device_ids):  
    world_size = len(device_ids)
    device_id = device_ids[rank]
    device = f"cuda:{device_id}"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    os.chdir(os.path.join(os.path.dirname(__file__), "../diffusion_policy"))
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, rank, world_size, device_id, device)
    workspace.load_checkpoint(path=workspace.get_checkpoint_path(), exclude_keys=['optimizer'], include_keys=[])
    # get policy from workspace
    policy = workspace.model.module
    if cfg.training.use_ema:
        policy = workspace.ema_model.module

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    To = cfg.n_obs_steps
    policy.set_normalizer(normalizer)

    policy.to(device)
    policy.eval()

    exp_name = cfg.output_dir.split('/')[-1]

    baseline_model = get_baseline_model(f'/home/yuwenye/project/human-in-the-loop/baseline_data/outputs/{exp_name}/baseline.pt', device=device)
    replay_buffer = ReplayBuffer.copy_from_path(f'/home/yuwenye/project/human-in-the-loop/baseline_data/rollout/{exp_name}/replay_buffer.zarr')
    episode_length = get_episode_length(replay_buffer)
    logpZO_list = []
    for idx in range(replay_buffer.n_episodes): 
        logpZO = get_episode_and_return_logpZO(dataset, policy, baseline_model, replay_buffer, idx, episode_length, device=device, To=To)
        # print(f"Current episode logpZO: {logpZO.reshape(-1)}")
        logpZO_list.append(logpZO.reshape(-1))
    logpZO_list = torch.stack(logpZO_list, dim=0).detach().cpu().numpy() # (n_episodes, episode_length)
    n_train = int(logpZO_list.shape[0] * 0.3)

    predictor = FunctionalPredictor(modulation_type=ModulationType.Tfunc, regression_type=RegressionType.Mean)
    target_traj = predictor.get_one_sided_prediction_band(logpZO_list[:n_train], logpZO_list[-n_train:], alpha=0.05, lower_bound=False).flatten()
    np.savez(f"/home/yuwenye/project/human-in-the-loop/baseline_data/outputs/{exp_name}/target_traj.npz", target_traj=target_traj)
    torch.save(normalizer, f"/home/yuwenye/project/human-in-the-loop/baseline_data/outputs/{exp_name}/normalizer.pt")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    with hydra.initialize(config_path="../diffusion_policy/diffusion_policy/config"):
        cfg = hydra.compose(config_name=sys.argv[1])

    device_ids = [int(x) for x in cfg.device_ids.split(",")]
    os.environ["MASTER_ADDR"] = "localhost"
    port = 30003
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
    if len(device_ids) == 1:
        main(0, cfg, device_ids)
    elif len(device_ids) > 1:
        OmegaConf.resolve(cfg)
        processes = []
        for rank in range(len(device_ids)):
            p = Process(target=main, args=(rank, cfg, device_ids))
            p.start()     
            processes.append(p)
        for p in processes:
            p.join()
        # torch.multiprocessing.spawn(main, args=(cfg, device_ids), nprocs=len(device_ids), join=True)