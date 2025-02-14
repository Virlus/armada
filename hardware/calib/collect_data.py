from my_device.robot import FlexivRobot
from my_device.camera import CameraD400
import time
import numpy as np
from PIL import Image
import os
import cv2
from scipy.spatial.transform import Rotation as R
from my_device.keyboard import Keyboard
def checkChessboard(win_name, color_image, shape=(11,8)):
    color_image = color_image.copy()
    flag, corners = cv2.findChessboardCorners(color_image, shape, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if flag:
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        img = cv2.drawChessboardCorners(color_image, shape, corners2, flag)
    else:
        img = color_image
    cv2.imshow(win_name, img)
    return flag

if __name__ == '__main__':
    cam = CameraD400('135122075425')
    # cam = CameraD400('104122061850')
    dir = os.path.join("/home/mzc/calib","test")
    os.makedirs(dir, exist_ok=True)
    k = Keyboard()
    for i in range(30):
        # time.sleep(0.1)
        while True:
            k.done = False
            data = None
            while not k.done:
                curr_time = int(time.time() * 1000)
                color_image, depth_image = cam.get_data()
                flag = checkChessboard('color', color_image)
                if flag:
                    data = curr_time, color_image, depth_image
                cv2.waitKey(100)
            if data is not None:
                break
        print(i, 'saved')
        curr_time, color_image, depth_image = data
        Image.fromarray(color_image).save(os.path.join(dir, f'{curr_time}c.png'))
        Image.fromarray(depth_image).save(os.path.join(dir, f'{curr_time}d.png'))
