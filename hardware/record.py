import time
import os
import json
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import shutil
import multiprocessing
from typing import List

from my_device.camera import CameraD400
from my_device.robot import FlexivRobot, FlexivGripper
from my_device.sigma import Sigma7
from my_device.keyboard import Keyboard

camera_serial = ["038522063145", "104422070044"]

path_prefix = '/mnt/workspace/DP/test/train'

def save_data(
        color_image, 
        depth_image, 
        color_dir,
        depth_dir,):
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    Image.fromarray(color_image).save(color_dir)
    Image.fromarray(depth_image).save(depth_dir)

def record(robot:FlexivRobot, gripper:FlexivGripper, cameras:List[CameraD400], sigma:Sigma7, keyboard: Keyboard, pool: multiprocessing.Pool):
    start_time = int(time.time() * 1000) # doesn't matter
    
    demo_path = os.path.join(path_prefix, f'{start_time}')
    os.makedirs(demo_path)
    # print(camera.getIntrinsics())
    # color_image, depth_image = camera.get_data()
    # color_image: 460,640,3  0~255
    # depth_image: 460,640  0~4681 [mm]
    
    cam_path = [os.path.join(demo_path, "cam_{}".format(s)) for s in camera_serial]
    color_dir = [os.path.join(path, 'color') for path in cam_path]
    depth_dir = [os.path.join(path, 'depth') for path in cam_path]
    for path in cam_path:
        os.mkdir(path)
    for path in color_dir:
        os.mkdir(path)
    for path in depth_dir:
        os.mkdir(path)

    tcp_dir = os.path.join(demo_path, 'tcp')
    joint_dir = os.path.join(demo_path, 'joint')
    action_dir = os.path.join(demo_path, 'action')
    gripper_dir = os.path.join(demo_path, 'gripper_command')
    
    os.mkdir(tcp_dir)
    os.mkdir(joint_dir)
    os.mkdir(gripper_dir)
    os.mkdir(action_dir)

    with open(os.path.join(demo_path, "timestamp.txt"), "w") as f:
        f.write('2')

    keyboard.start = False
    keyboard.discard = False
    keyboard.finish = False
    cnt = 0
    start_time = None
    while not keyboard.quit and not keyboard.discard and not keyboard.finish:
        # time.sleep(0.1)
        curr_time = int(time.time() * 1000)
        cam_data = []
        for camera in cameras:
            color_image, depth_image = camera.get_data()
            cam_data.append((color_image, depth_image))
        tcpPose, jointPose, _, _ = robot.get_robot_state()
        
        diff_p, diff_r, width = sigma.get_control()
        # print(width)
        diff_p = diff_p + robot.init_pose[:3]
        diff_r = R.from_quat(robot.init_pose[3:]) * diff_r
        # Send command.
        robot.send_tcp_pose(np.concatenate((diff_p,diff_r.as_quat()), 0))
        gripper.move_from_sigma(width)
        gripper_width = gripper.max_width * width / 1000
        if not keyboard.start:
            continue
        cnt += 1
        start_time = time.time() if start_time is None else start_time
        for (color_image, depth_image), color_path, depth_path in zip(cam_data, color_dir, depth_dir):
            pool.apply_async(save_data, args=(
                color_image.copy(), 
                depth_image.copy(), 
                os.path.join(color_path, f'{curr_time}.png'), 
                os.path.join(depth_path, f'{curr_time}.png'))
            )
        # TCP: 7-dimensional
        np.save(os.path.join(tcp_dir, f'{curr_time}.npy'), tcpPose)
        # joint
        np.save(os.path.join(joint_dir, f'{curr_time}.npy'), jointPose)
        # gripper width
        np.save(os.path.join(gripper_dir, f'{curr_time}.npy'), [gripper_width])
        np.save(os.path.join(action_dir, f'{curr_time}.npy'), np.concatenate((diff_p,diff_r.as_quat())))
    if not keyboard.start or keyboard.quit or keyboard.discard:
        print('WARNING: discard the demo!')
        time.sleep(5)
        shutil.rmtree(demo_path)
        return
    print('saved:', demo_path, 'fps:', cnt / (time.time()-start_time + 1e-6))
    meta = {'finish_time': int(time.time() * 1000)}
    with open(os.path.join(demo_path, "metadata.json"), "w") as f:
        json.dump(meta, f)

def main():
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    camera = [CameraD400(s) for s in camera_serial]
    sigma = Sigma7()
    keyboard = Keyboard()
    pool = multiprocessing.Pool(16)
    while not keyboard.quit:
        print("start recording...")
        record(robot, gripper, camera, sigma, keyboard, pool)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()