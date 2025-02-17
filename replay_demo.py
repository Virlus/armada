from hardware.my_device.robot import FlexivRobot, FlexivGripper
from third_party.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
import argparse
import time

def main(args):
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    replay_buffer = ReplayBuffer.copy_from_path(args.demo_path, keys=['wrist_cam', 'side_cam', 'joint_pos', 'action'])

    for i in range(replay_buffer.n_episodes):
        robot.send_tcp_pose(robot.init_pose)
        gripper.move(gripper.max_width)
        time.sleep(2)
        for j in range(0 if i == 0 else replay_buffer.episode_ends[i-1], replay_buffer.episode_ends[i]):
            start_time = time.time()
            robot.send_tcp_pose(replay_buffer['action'][j, :7])
            gripper.move(replay_buffer['action'][j, 7])
            time.sleep(max(1 / 10 - (time.time() - start_time), 0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--demo_path', type=str, default='/mnt/workspace/DP/0217_stack_cups/replay_buffer.zarr')
    args = parser.parse_args()
    main(args)