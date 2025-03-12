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


def get_interpolated_array(num, start, end):
    weight = np.linspace(0, 1, num=num, endpoint=False)
    return np.array([start + w * (end - start) for w in weight])


if __name__ == '__main__':
    robot = FlexivRobot()
    # robot.robot.setMode(robot.mode.NRT_PLAN_EXECUTION)
    # robot.robot.executePlan("PLAN-FreeDriveAuto")

    key_poses = np.array([
        [0.6,0,0.2,0,0,1,0],
        [0.43033236265182495, 0.07971877604722977, 0.42104658484458923, -0.029554568231105804, -0.282661497592926, 0.9364845752716064, -0.20548869669437408],
        [0.39390870928764343, -0.09365829825401306, 0.28368425369262695, 0.21174144744873047, -0.1830323487520218, 0.9599466919898987, 0.012924541719257832],
        [0.6481845378875732, -0.13726219534873962, 0.33488303422927856, -0.04586920887231827, -0.29011276364326477, 0.9538244605064392, -0.0628449097275734]
    ])

    all_poses = [get_interpolated_array(10, key_poses[i], key_poses[i+1]) for i in range(len(key_poses)-1)]
    all_poses = np.concatenate(all_poses, axis=0)

    cam_hand = CameraD400("104422070044", is_calib=True)
    cam_side = CameraD400("038522063145")
    dir = os.path.join("./workspace/flexiv/calib/out","cali_side")
    os.makedirs(dir)
    dir_hand = os.path.join(dir, 'hand')
    dir_side = os.path.join(dir, 'side')
    os.mkdir(dir_hand)
    os.mkdir(dir_side)
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
                color_image_hand, depth_image_hand = cam_hand.get_data()
                color_image, depth_image = cam_side.get_data()
                tcpPose, jointPose, tcpVel, jointVel = robot.get_robot_state()
                flag1, result_img = checkChessboard(color_image)
                flag2, result_img_hand = checkChessboard(color_image_hand)
                cv2.imshow('side', result_img)
                cv2.imshow('hand', result_img_hand)
                cv2.waitKey(100)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                result_img_hand = cv2.cvtColor(result_img_hand, cv2.COLOR_BGR2RGB)

                if flag1 and flag2:
                    data = curr_time, color_image_hand, depth_image_hand, color_image, depth_image, result_img, result_img_hand, tcpPose, jointPose
            if data is not None:
                break
            else:
                print('no chessboard found')
        
        curr_time, color_image_hand, depth_image_hand, color_image, depth_image, result_img, result_img_hand, tcpPose, jointPose = data
        print(i, 'saved')
        Image.fromarray(result_img_hand).save(os.path.join(dir_hand, f'{curr_time}r.png'))
        Image.fromarray(color_image_hand).save(os.path.join(dir_hand, f'{curr_time}c.png'))
        Image.fromarray(depth_image_hand).save(os.path.join(dir_hand, f'{curr_time}d.png'))
        
        Image.fromarray(result_img).save(os.path.join(dir_side, f'{curr_time}r.png'))
        Image.fromarray(color_image).save(os.path.join(dir_side, f'{curr_time}c.png'))
        Image.fromarray(depth_image).save(os.path.join(dir_side, f'{curr_time}d.png'))
        np.savetxt(os.path.join(dir, f'{curr_time}.txt'), tcpPose)

       

