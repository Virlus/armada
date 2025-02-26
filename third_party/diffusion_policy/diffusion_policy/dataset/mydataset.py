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
            n_obs_steps=1
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['wrist_cam', 'side_cam', 'joint_pos', 'action'])
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
        data = {
            'action': self.replay_buffer['action'],
            'qpos': self.replay_buffer['joint_pos'][...,:7]
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
            base_pose = action_sample[self.n_obs_steps-2, :7]
            base_rot = R.from_quat(base_pose[3:][[1,2,3,0]])
            base_translate = base_pose[:3]
            for i in range(self.n_obs_steps-1, self.horizon):
                curr_rot = R.from_quat(action_sample[i, [4,5,6,3]])
                curr_translate = action_sample[i, :3]
                action_sample[i, :3] = (curr_translate - base_translate) @ base_rot.as_matrix()
                action_sample[i, 3:7] = (base_rot.inv() * curr_rot).as_quat()[[3,0,1,2]]
        else:
            action_sample = sample['action']

        data = {
            'obs': {
                'wrist_img': wrist_img, # T, 3, 480, 640
                'side_img': side_img, # T, 3, 480, 640
                'qpos': qpos, # T, 7
            },
            'action': action_sample.astype(np.float32) # T, 8
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
    zarr_path = os.path.expanduser('/mnt/workspace/DP/0226_dagger/replay_buffer.zarr')
    dataset = MyDataset(zarr_path, horizon=16, rel_ee_pose=False, n_obs_steps=2)
    for i in range(len(dataset)):
        img = (dataset[i]['obs']['side_img'][0].detach().cpu().numpy() * 255.0).astype(np.uint8)
        img = np.moveaxis(img, 0, -1)
        Image.fromarray(img).save('/home/yuwenye/Desktop/debug/{}.png'.format(i))

if __name__ == '__main__':
    test()