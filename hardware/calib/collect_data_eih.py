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
from scipy.spatial.transform.rotation import Rotation as R
from my_device.keyboard import Keyboard

from calib.utils import checkChessboard

# def checkChessboard(win_name, color_image, shape=(11,8)):
#     flag, corners = cv2.findChessboardCorners(color_image, shape, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
#     if flag:
#         gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
#         img = cv2.drawChessboardCorners(color_image, shape, corners2, flag)
#     else:
#         img = color_image
#     cv2.imshow(win_name, img)
#     return flag

def get_interpolated_array(num, start, end):
    weight = np.linspace(0, 1, num=num, endpoint=False)
    return np.array([start + w * (end - start) for w in weight])

key_poses = np.array([
    [0.6,0,0.2,0,0,1,0],
    [0.43033236265182495, 0.07971877604722977, 0.42104658484458923, -0.029554568231105804, -0.282661497592926, 0.9364845752716064, -0.20548869669437408],
    [0.39390870928764343, -0.09365829825401306, 0.28368425369262695, 0.21174144744873047, -0.1830323487520218, 0.9599466919898987, 0.012924541719257832],
    [0.6481845378875732, -0.13726219534873962, 0.33488303422927856, -0.04586920887231827, -0.29011276364326477, 0.9538244605064392, -0.0628449097275734]
])

all_poses = [get_interpolated_array(10, key_poses[i], key_poses[i+1]) for i in range(len(key_poses)-1)]
all_poses = np.concatenate(all_poses, axis=0)

if __name__ == '__main__':
    robot = FlexivRobot()
    # robot.robot.setMode(robot.mode.NRT_PLAN_EXECUTION)
    # robot.robot.executePlan("PLAN-FreeDriveAuto")

    cam = CameraD400("135122070361", is_calib=True)
    dir = os.path.join("./workspace/flexiv/calib/out","cali_hand")
    os.makedirs(dir)
    k = Keyboard()
    for i in range(30):
        robot.send_tcp_pose(all_poses[i])
        time.sleep(2)
        # time.sleep(0.1)
        while True:
            k.finish = False
            data = None
            while not k.finish:
                curr_time = int(time.time() * 1000)
                color_image, depth_image = cam.get_data()
                tcpPose, jointPose, tcpVel, jointVel = robot.get_robot_state()
                flag, result_img = checkChessboard(color_image)
                cv2.imshow('color', result_img)
                cv2.waitKey(100)
                if flag:
                    data = curr_time, tcpPose, jointPose, color_image, depth_image
            if data is not None:
                break
        curr_time, tcpPose, jointPose, color_image, depth_image = data
        print(i, 'saved')
        Image.fromarray(color_image).save(os.path.join(dir, f'{curr_time}c.png'))
        Image.fromarray(depth_image).save(os.path.join(dir, f'{curr_time}d.png'))
        np.savetxt(os.path.join(dir, f'{curr_time}t.txt'), tcpPose)
        np.savetxt(os.path.join(dir, f'{curr_time}j.txt'), jointPose)
