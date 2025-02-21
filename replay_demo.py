from hardware.my_device.robot import FlexivRobot, FlexivGripper
from third_party.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from scipy.spatial.transform import Rotation as R
import argparse
import numpy as np
import time

def main(args):
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    replay_buffer = ReplayBuffer.copy_from_path(args.demo_path, keys=['wrist_cam', 'side_cam', 'joint_pos', 'action'])

    for i in range(replay_buffer.n_episodes-1, replay_buffer.n_episodes):
        robot.send_tcp_pose(robot.init_pose)
        gripper.move(gripper.max_width)
        last_p = robot.init_pose[:3]
        last_r = R.from_quat(robot.init_pose[3:])
        time.sleep(2)
        for j in range(0 if i == 0 else replay_buffer.episode_ends[i-1], replay_buffer.episode_ends[i]):
            start_time = time.time()
            curr_p = last_p + replay_buffer['action'][j, :3]
            curr_r = last_r * R.from_quat(replay_buffer['action'][j, 3:7])
            robot.send_tcp_pose(np.concatenate((curr_p, curr_r.as_quat()), 0))
            last_p = curr_p
            last_r = curr_r
            # robot.send_tcp_pose(replay_buffer['action'][j, :7])
            if replay_buffer['action'][j, 7] == 0:
                gripper.move(gripper.max_width)
            else:
                gripper.move(0)
            time.sleep(max(1 / 10 - (time.time() - start_time), 0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--demo_path', type=str, default='/mnt/workspace/DP/0221_test/replay_buffer.zarr')
    args = parser.parse_args()
    main(args)