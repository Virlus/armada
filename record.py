import time
import os
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from typing import List
import argparse
import torch
from torchvision.transforms import Compose, Resize
from torchvision.transforms import InterpolationMode

from hardware.my_device.camera import CameraD400
from hardware.my_device.robot import FlexivRobot, FlexivGripper
from hardware.my_device.sigma import Sigma7
from hardware.my_device.keyboard import Keyboard

from third_party.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer

camera_serial = ["038522063145", "104422070044"]


def record(replay_buffer:ReplayBuffer, robot:FlexivRobot, gripper:FlexivGripper, cameras:List[CameraD400], sigma:Sigma7, \
           keyboard: Keyboard, image_processor: Compose):
    start_time = int(time.time() * 1000) # doesn't matter
    
    # color_image, depth_image = camera.get_data()
    # color_image: 480,640,3  0~255
    # depth_image: 480,640  0~4681 [mm]

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
    
    # Relative ee pose as action space
    last_p, last_r, width = sigma.get_control()
    base_p = last_p
    base_r = last_r

    while not keyboard.quit and not keyboard.discard and not keyboard.finish:
        time.sleep(max(0.1 - (time.time() - start_time), 0))
        start_time = time.time()
        cam_data = []
        for camera in cameras:
            color_image, depth_image = camera.get_data()
            cam_data.append((color_image, depth_image))
        tcpPose, jointPose, _, _ = robot.get_robot_state()
        # gripper_state = gripper.get_gripper_state()
        
        diff_p, diff_r, width = sigma.get_control()
        curr_p_action = diff_p - last_p
        curr_r_action = last_r.inv() * diff_r
        last_p = diff_p
        last_r = diff_r
        curr_p = diff_p - base_p + robot.init_pose[:3]
        curr_r = R.from_quat(robot.init_pose[3:]) * base_r.inv() * diff_r
        # Send command.
        robot.send_tcp_pose(np.concatenate((curr_p,curr_r.as_quat()), 0))
        gripper.move_from_sigma(width)
        # gripper_width = gripper.max_width * width / 1000
        gripper_action = 1 if width < 500 else 0
        if not keyboard.start:
            continue

        # Initialize at the beginning of the episode
        if cnt == 0:
            robot.send_tcp_pose(robot.init_pose)
            time.sleep(1.5)
            gripper.move(gripper.max_width)
            time.sleep(0.5)
            base_p, base_r, _ = sigma.get_control()
            last_p = base_p
            last_r = base_r
            cnt += 1
            print("Episode start!")

        wrist_image = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1][0].copy(), cv2.COLOR_BGR2RGB)).\
                                      permute(2,0,1)).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        side_image = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0][0].copy(), cv2.COLOR_BGR2RGB)).\
                                      permute(2,0,1)).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        wrist_cam.append(wrist_image)
        side_cam.append(side_image)
        tcp_pose.append(tcpPose)
        joint_pos.append(jointPose)
        action.append(np.concatenate((curr_p_action, curr_r_action.as_quat(), [gripper_action])))

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
    # Image processing
    img_res = args.resolution
    BICUBIC = InterpolationMode.BICUBIC
    image_processor = Compose([Resize(img_res, interpolation=BICUBIC)])
    while not keyboard.quit:
        print("start recording...")
        record(replay_buffer, robot, gripper, camera, sigma, keyboard, image_processor)
        if not keyboard.quit:
            print("reset the environment...")
            time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='/mnt/workspace/DP/0219_PnP_fixed_init')
    parser.add_argument('-res', '--resolution', nargs='+', type=int)
    args = parser.parse_args()
    main(args)