import os
import argparse
from third_party.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer


def main(args):
    save_buffer = ReplayBuffer.create_empty_numpy()
    for file in os.listdir(args.dataset_path):
        if args.filter in file:
            file_path = os.path.join(args.dataset_path, file, 'replay_buffer.zarr')
            replay_buffer = ReplayBuffer.copy_from_path(file_path, keys=None)
            for i in range(replay_buffer.n_episodes):
                episode = replay_buffer.get_episode(i)
                save_buffer.add_episode(episode)
    save_buffer.save_to_path(os.path.join(args.dataset_path, args.save, 'replay_buffer.zarr'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--dataset_path', type=str, required=True)
    parser.add_argument('-f', '--filter', type=str, required=True)
    parser.add_argument('-s', '--save', type=str, required=True)
    args = parser.parse_args()
    
    main(args)