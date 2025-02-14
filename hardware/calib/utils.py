import numpy as np
import cv2

def _inpaint(img, missing_value=0):
        '''
        pip opencv-python == 3.4.8.29
        :param image:
        :param roi: [x0,y0,x1,y1]
        :param missing_value:
        :return:
        '''
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(img).max()
        if scale < 1e-3:
            pdb.set_trace()
        img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        img = img[1:-1, 1:-1]
        img = img * scale
        return img

def getXYZRGB(color, depth, robot_pose, camee_pose, camIntrinsics, inpaint=True):
    '''

    :param color:
    :param depth:
    :param robot_pose: array 4*4
    :param camee_pose: array 4*4
    :param camIntrinsics: array 3*3
    :param inpaint: bool
    :return: xyzrgb
    '''
    import open3d as o3d
    if inpaint:
        depth = _inpaint(depth)
    color_image = o3d.geometry.Image(color)
    depth_image = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, convert_rgb_to_intensity=False)

    fx, fy, cx, cy = camIntrinsics[0, 0], camIntrinsics[1, 1], camIntrinsics[0, 2], camIntrinsics[1, 2]
    width, height = color.shape[1], color.shape[0]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    world_pose = np.dot(robot_pose, camee_pose)
    pcd.transform(world_pose)

    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    
    xyzrgb = np.hstack((xyz, rgb))
    return xyzrgb

def checkChessboard(color_image, shape=(11,8)):
    color_image = color_image.copy()
    flag, corners = cv2.findChessboardCorners(color_image, shape, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if flag:
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        img = cv2.drawChessboardCorners(color_image, shape, corners2, flag)
    else:
        img = color_image
    return flag, img
