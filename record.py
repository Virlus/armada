import time
import os
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from typing import List
import argparse

from hardware.my_device.camera import CameraD400
from hardware.my_device.robot import FlexivRobot, FlexivGripper
from hardware.my_device.sigma import Sigma7
from hardware.my_device.keyboard import Keyboard

from third_party.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer

camera_serial = ["038522063145", "104422070044"]

def save_data(
        color_image, 
        depth_image, 
        color_dir,
        depth_dir,):
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    Image.fromarray(color_image).save(color_dir)
    Image.fromarray(depth_image).save(depth_dir)

def record(replay_buffer:ReplayBuffer, robot:FlexivRobot, gripper:FlexivGripper, cameras:List[CameraD400], sigma:Sigma7, \
           keyboard: Keyboard):
    start_time = int(time.time() * 1000) # doesn't matter
    
    # color_image, depth_image = camera.get_data()
    # color_image: 460,640,3  0~255
    # depth_image: 460,640  0~4681 [mm]

    tcp_pose = []
    joint_pos = []
    action = []
    wrist_cam = []
    side_cam = []

    keyboard.start = False
    keyboard.discard = False
    keyboard.finish = False
    cnt = 0
    start_time = time.time()
    while not keyboard.quit and not keyboard.discard and not keyboard.finish:
        time.sleep(max(0.1 - (time.time() - start_time), 0))
        start_time = time.time()
        cam_data = []
        for camera in cameras:
            color_image, depth_image = camera.get_data()
            cam_data.append((color_image, depth_image))
        tcpPose, jointPose, _, _ = robot.get_robot_state()
        
        diff_p, diff_r, width = sigma.get_control()
        diff_p = diff_p + robot.init_pose[:3]
        diff_r = R.from_quat(robot.init_pose[3:]) * diff_r
        # Send command.
        robot.send_tcp_pose(np.concatenate((diff_p,diff_r.as_quat()), 0))
        gripper.move_from_sigma(width)
        gripper_width = gripper.max_width * width / 1000
        if not keyboard.start:
            continue
        cnt += 1
        wrist_image = cv2.cvtColor(cam_data[1][0].copy(), cv2.COLOR_BGR2RGB)
        side_image = cv2.cvtColor(cam_data[0][0].copy(), cv2.COLOR_BGR2RGB)
        wrist_cam.append(wrist_image)
        side_cam.append(side_image)
        tcp_pose.append(tcpPose)
        joint_pos.append(jointPose)
        action.append(np.concatenate((diff_p,diff_r.as_quat(), [gripper_width])))

    if not keyboard.start or keyboard.quit or keyboard.discard:
        print('WARNING: discard the demo!')
        time.sleep(5)
        return
    
    episode = dict()
    episode['wrist_cam'] = np.stack(wrist_cam, axis=0)
    episode['side_cam'] = np.stack(side_cam, axis=0)
    episode['tcp_pose'] = np.stack(tcp_pose, axis=0)
    episode['joint_pos'] = np.stack(joint_pos, axis=0)
    episode['action'] = np.stack(action, axis=0)
    replay_buffer.add_episode(episode, compressors='disk')
    episode_id = replay_buffer.n_episodes - 1
    print('Saved episode ', episode_id)


def main(args):
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    camera = [CameraD400(s) for s in camera_serial]
    sigma = Sigma7()
    keyboard = Keyboard()
    zarr_path = os.path.join(args.output, 'replay_buffer.zarr')
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='a')
    while not keyboard.quit:
        print("start recording...")
        record(replay_buffer, robot, gripper, camera, sigma, keyboard)
        if not keyboard.quit:
            print("reset the environment...")
            time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='/mnt/workspace/DP/0217_stack_cups')
    args = parser.parse_args()
    main(args)