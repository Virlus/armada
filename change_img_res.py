import numpy as np
import torch
from torchvision.transforms import Compose, Resize
from torchvision.transforms import InterpolationMode
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname((__file__)), 'third_party/diffusion_policy'))

from third_party.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer


def main(args):
    replay_buffer = ReplayBuffer.copy_from_path(args.dataset_path, keys=['wrist_cam', 'side_cam', \
        'joint_pos', 'action', 'tcp_pose'])
    save_buffer = ReplayBuffer.create_empty_numpy()
    
    BICUBIC = InterpolationMode.BICUBIC
    image_processor = Compose([Resize(args.resolution, interpolation=BICUBIC)])
    
    for i in range(replay_buffer.n_episodes):
        episode = replay_buffer.get_episode(i)
        wrist_cam = np.moveaxis(image_processor(torch.from_numpy(episode['wrist_cam']).permute(0, 3, 1, 2)).detach().cpu().numpy(), 1, -1)
        side_cam = np.moveaxis(image_processor(torch.from_numpy(episode['side_cam']).permute(0, 3, 1, 2)).detach().cpu().numpy(), 1, -1)
        
        episode['wrist_cam'] = wrist_cam
        episode['side_cam'] = side_cam
        
        save_buffer.add_episode(episode)
        
    save_buffer.save_to_path(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', '--resolution', nargs='+', type=int)
    parser.add_argument('-p', '--dataset_path', type=str, required=True)
    parser.add_argument('-save', '--save_path', type=str, required=True)
    args = parser.parse_args()
    
    main(args)