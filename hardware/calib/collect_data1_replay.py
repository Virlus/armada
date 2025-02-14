import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../flexiv_rdk/lib_py"))
from my_device.robot import FlexivRobot
from my_device.camera import CameraD400
import time
import numpy as np
from PIL import Image
import os
import cv2
from scipy.spatial.transform import Rotation as R, Slerp
from my_device.keyboard import Keyboard
import glob

def interp(joints, k):
    n = len(joints)
    ret = [joints[0]]
    for i in range(n-1):
        for j in range(k):
            ret.append( (k-j-1)/k * joints[i] + (j+1)/k *joints[i+1])
    return ret
    
from calib.utils import checkChessboard

if __name__ == '__main__':
    robot = FlexivRobot(default_pose=(0.6,0,0.2,0,0.5**0.5,0.5**0.5,0))
    cam = CameraD400('135122079702', is_calib=True)
    jointList = glob.glob("/home/jinyang/workspace/flexiv/calib/out/old/cali_hand_old/*j.txt")
    jointList.sort()
    dir = os.path.join("/home/jinyang/workspace/flexiv/calib/out","cali_hand")
    os.makedirs(dir)
    joints = [np.loadtxt(filename) for filename in jointList]
    # joints = interp(joints, 5)

    for i, joint in enumerate(joints):
        print(joint)
        robot.send_joint_pose(joint)
        while True:
            time.sleep(0.5)
            tcpPose, jointPose, tcpVel, jointVel = robot.get_robot_state()
            diff = np.linalg.norm(np.array(jointPose) - np.array(joint))
            if (diff < 0.01):
                break
        time.sleep(0.5)
        curr_time = int(time.time() * 1000)
        color_image, depth_image = cam.get_data()
        print(color_image.shape)
        tcpPose, jointPose, tcpVel, jointVel = robot.get_robot_state()
        flag, result_img = checkChessboard(color_image)
        cv2.imshow('color', result_img)
        cv2.waitKey(100)
        if flag:
            print(i, 'saved')
            Image.fromarray(result_img).save(os.path.join(dir, f'{curr_time}r.png'))
            Image.fromarray(color_image).save(os.path.join(dir, f'{curr_time}c.png'))
            Image.fromarray(depth_image).save(os.path.join(dir, f'{curr_time}d.png'))
            np.savetxt(os.path.join(dir, f'{curr_time}t.txt'), tcpPose)
            np.savetxt(os.path.join(dir, f'{curr_time}j.txt'), jointPose)
        else:
            print('chessboard not found')
