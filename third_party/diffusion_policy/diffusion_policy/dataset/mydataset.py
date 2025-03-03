import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from scipy.spatial.transform import Rotation as R

class MyDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            rel_ee_pose=False,
            n_obs_steps=1,
            shape_meta=None,
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['wrist_cam', 'side_cam', 'joint_pos', 'action', 'tcp_pose'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.rel_ee_pose = rel_ee_pose
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.action_dim = self.shape_meta['action']['shape'][0]

        self.action_rot_transformer = None
        self.obs_rot_transformer = None
        # Check if there's need for transforming rotation representation
        if 'rotation_rep' in shape_meta['action']:
            self.action_rot_transformer = RotationTransformer(from_rep='quaternion', to_rep=shape_meta['action']['rotation_rep'])
        if 'ee_pose' in shape_meta['obs']:
            self.ee_pose_dim = self.shape_meta['obs']['ee_pose']['shape'][0]
            if 'rotation_rep' in shape_meta['obs']['ee_pose']:
                self.obs_rot_transformer = RotationTransformer(from_rep='quaternion', to_rep=shape_meta['obs']['ee_pose']['rotation_rep'])

        # Currently we only support multiple observation steps
        assert self.n_obs_steps > 1

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        if self.rel_ee_pose:
            rel_action_buffer = []
            # rel_obs_buffer = []
            for episode_id in range(self.replay_buffer.n_episodes):
                for step in range(0 if episode_id == 0 else self.replay_buffer.episode_ends[episode_id-1], \
                                  self.replay_buffer.episode_ends[episode_id] - self.horizon + 1):
                    # Relative action space
                    base_pose = self.replay_buffer['action'][step, :7]
                    base_rot = R.from_quat(base_pose[np.newaxis, [4,5,6,3]].repeat(self.horizon-(self.n_obs_steps-1), axis=0))
                    base_translate = base_pose[np.newaxis, :3].repeat(self.horizon-(self.n_obs_steps-1), axis=0)
                    curr_rot = R.from_quat(self.replay_buffer['action'][step+self.n_obs_steps-1:step+self.horizon, [4,5,6,3]])
                    curr_translate = self.replay_buffer['action'][step+self.n_obs_steps-1:step+self.horizon, :3]
                    rel_action = np.zeros((self.horizon-(self.n_obs_steps-1), self.action_dim))
                    # rel_action[:, :3] = np.matmul(curr_translate - base_translate, base_rot[0].as_matrix())
                    rel_action[:, :3] = curr_translate - base_translate
                    rel_rot = (base_rot.inv() * curr_rot).as_quat()[:, [3,0,1,2]]
                    if self.action_rot_transformer is not None:
                        rel_action[:, 3:self.action_dim-1] = self.action_rot_transformer.forward(rel_rot)
                    else:
                        rel_action[:, 3:7] = rel_rot
                    rel_action[:, -1] = self.replay_buffer['action'][step+self.n_obs_steps-1:step+self.horizon, 7]
                    rel_action_buffer.append(rel_action)

                    # # Relative ee pose obs
                    # prev_timesteps = np.arange(self.n_obs_steps-1)
                    # curr_timesteps = np.arange(self.n_obs_steps-1) + 1
                    # prev_pose = self.replay_buffer['tcp_pose'][step+prev_timesteps, :]
                    # curr_pose = self.replay_buffer['tcp_pose'][step+curr_timesteps, :]
                    # prev_rot = R.from_quat(self.replay_buffer['tcp_pose'][step+prev_timesteps, :][:, [4,5,6,3]]) 
                    # curr_rot = R.from_quat(self.replay_buffer['tcp_pose'][step+curr_timesteps, :][:, [4,5,6,3]])
                    # rel_translate = np.matmul((curr_pose[:, :3] - prev_pose[:, :3])[:, np.newaxis, :], prev_rot.as_matrix()).squeeze(1)
                    # rel_rotation = prev_rot.inv() * curr_rot
                    # rel_rotation6d = rel_rotation.as_matrix()[:, :2, :].reshape(-1, 6)
                    # rel_ee_pose = np.concatenate((rel_translate, rel_rotation6d), axis=1)
                    # rel_obs_buffer.append(rel_ee_pose)
            rel_action_dataset = np.concatenate(rel_action_buffer, axis=0)
            # rel_obs_dataset = np.concatenate(rel_obs_buffer, axis=0)
            data = {
                'action': rel_action_dataset,
                'qpos': self.replay_buffer['joint_pos'][...,:7]
                # 'rel_ee_pose': rel_obs_dataset,
            }
        else:
            action_sample = self.replay_buffer['action']
            action_processed = np.zeros((action_sample.shape[0], self.action_dim))
            action_processed[:, :3] = action_sample[:, :3]
            action_rotation = R.from_quat(action_sample[:, [4,5,6,3]]).as_quat()[:, [3,0,1,2]]
            if self.action_rot_transformer is not None:
                action_processed[:, 3:self.action_dim-1] = self.action_rot_transformer.forward(action_rotation)
            else:
                action_processed[:, 3:7] = action_rotation
            action_processed[:, -1] = action_sample[:, 7]

            if 'ee_pose' in self.shape_meta['obs']:
                tcp_pose_sample = self.replay_buffer['tcp_pose']
                ee_pose = np.zeros((tcp_pose_sample.shape[0], self.ee_pose_dim))
                ee_pose[:, :3] = tcp_pose_sample[:, :3]
                rel_obs_ee = R.from_quat(tcp_pose_sample[:, [4,5,6,3]]).as_quat()[:, [3,0,1,2]]
                if self.obs_rot_transformer is not None:
                    ee_pose[:, 3:self.ee_pose_dim] = self.obs_rot_transformer.forward(rel_obs_ee)
                else:
                    ee_pose[:, 3:7] = rel_obs_ee
            
            # data = {
            #     'action': self.replay_buffer['action'],
            #     'qpos': self.replay_buffer['joint_pos'][...,:7]
            # }
            data = {
                'action': action_processed,
                'ee_pose': ee_pose
            }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['wrist_img'] = get_image_range_normalizer()
        normalizer['side_img'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        qpos = sample['joint_pos'][:,:7].astype(np.float32)
        wrist_img = np.moveaxis(sample['wrist_cam'],-1,1)/255
        side_img = np.moveaxis(sample['side_cam'],-1,1)/255

        if self.rel_ee_pose:
            action_sample = sample['action'].copy()
            # ee_pose_sample = sample['tcp_pose'].copy()
            # Compute the relative action between frames
            rel_action = np.zeros((self.horizon, self.action_dim))
            base_pose = action_sample[self.n_obs_steps-2, :7]
            base_rot = R.from_quat(base_pose[np.newaxis, [4,5,6,3]].repeat(self.horizon, axis=0))
            base_translate = base_pose[np.newaxis, :3].repeat(self.horizon, axis=0)
            curr_rot = R.from_quat(action_sample[:, [4,5,6,3]])
            curr_translate = action_sample[:, :3]

            # rel_action[:, :3] = np.matmul(curr_translate - base_translate, base_rot[0].as_matrix())
            rel_action[:, :3] = curr_translate - base_translate
            rel_rot = (base_rot.inv() * curr_rot).as_quat()[:, [3,0,1,2]]
            if self.action_rot_transformer is not None:
                rel_action[:, 3:self.action_dim-1] = self.action_rot_transformer.forward(rel_rot)
            else:
                rel_action[:, 3:7] = rel_rot
            rel_action[:, -1] = action_sample[:, 7]
            action_sample = rel_action
            # # Compute the relative ee pose as observations
            # prev_timesteps = np.arange(self.n_obs_steps-1)
            # curr_timesteps = np.arange(self.n_obs_steps-1) + 1
            # prev_pose = ee_pose_sample[prev_timesteps, :]
            # curr_pose = ee_pose_sample[curr_timesteps, :]
            # prev_rot = R.from_quat(ee_pose_sample[prev_timesteps, :][:, [4,5,6,3]]) 
            # curr_rot = R.from_quat(ee_pose_sample[curr_timesteps, :][:, [4,5,6,3]])
            # rel_translate = np.matmul((curr_pose[:, :3] - prev_pose[:, :3])[:, np.newaxis, :], prev_rot.as_matrix()).squeeze(1)
            # rel_rotation = prev_rot.inv() * curr_rot
            # rel_rotation6d = rel_rotation.as_matrix()[:, :2, :].reshape(-1, 6)
            # rel_ee_pose = np.concatenate((rel_translate, rel_rotation6d), axis=1)
            # rel_ee_pose = rel_ee_pose.reshape(-1)[np.newaxis, :].repeat(self.horizon, axis=0)
            data = {
                'obs': {
                    'wrist_img': wrist_img, # T, 3, 480, 640
                    'side_img': side_img, # T, 3, 480, 640
                    'qpos': qpos, # T, 7
                    # 'rel_ee_pose': rel_ee_pose,  # T, 9
                },
                'action': action_sample.astype(np.float32) # T, self.action_dim
            }
        else:
            action_sample = sample['action'].copy()
            action_processed = np.zeros((action_sample.shape[0], self.action_dim))
            action_processed[:, :3] = action_sample[:, :3]
            action_rotation = R.from_quat(action_sample[:, [4,5,6,3]]).as_quat()[:, [3,0,1,2]]
            if self.action_rot_transformer is not None:
                action_processed[:, 3:self.action_dim-1] = self.action_rot_transformer.forward(action_rotation)
            else:
                action_processed[:, 3:7] = action_rotation
            action_processed[:, -1] = action_sample[:, 7]

            if 'ee_pose' in self.shape_meta['obs']:
                tcp_pose_sample = sample['tcp_pose'].copy()
                ee_pose = np.zeros((tcp_pose_sample.shape[0], self.ee_pose_dim))
                ee_pose[:, :3] = tcp_pose_sample[:, :3]
                rel_ee_rot = R.from_quat(tcp_pose_sample[:, [4,5,6,3]]).as_quat()[:, [3,0,1,2]]
                if self.obs_rot_transformer is not None:
                    ee_pose[:, 3:self.ee_pose_dim] = self.obs_rot_transformer.forward(rel_ee_rot)
                else:
                    ee_pose[:, 3:7] = rel_ee_rot
            
            data = {
                'obs': {
                    'wrist_img': wrist_img, # T, 3, 480, 640
                    'side_img': side_img, # T, 3, 480, 640
                    # 'qpos': qpos, # T, 7
                    # 'rel_ee_pose': rel_ee_pose,  # T, 9
                    'ee_pose': ee_pose, # T, self.ee_pose_dim
                },
                # 'action': action_sample.astype(np.float32) # T, 8
                'action': action_processed.astype(np.float32), # T, self.action_dim
            }

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    from PIL import Image
    zarr_path = os.path.expanduser('/home/jinyang/yuwenye/human-in-the-loop/data/0225_abs_PnP/replay_buffer.zarr')
    shape_meta = {
        'obs':{
            'wrist_img':{
                'shape': [3, 240, 320],
                'type': 'rgb'
            },
            'side_img':{
                'shape': [3, 240, 320],
                'type': 'rgb'
            },
            'ee_pose':{
                'shape': [9],
                'type': 'low_dim',
                'rotation_rep': 'rotation_6d'
            },
        },
        'action':{
            'shape': [10],
            'rotation_rep': 'rotation_6d'
        }
    }
    dataset = MyDataset(zarr_path, horizon=16, rel_ee_pose=True, n_obs_steps=2, shape_meta=shape_meta)
    dataset.get_normalizer()
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    test()