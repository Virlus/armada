import time
import cv2
import numpy as np
import pygame
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import os, sys

class RealSenseCapture:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
    
    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return np.flip(color_image, -1).copy()

class CameraD400(object):
    def __init__(self, camera_id=0):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        ctx = rs.context()
        devices = ctx.query_devices()
        if camera_id >= len(devices):
            raise ValueError("Camera ID is out of range")
        self.config.enable_device(devices[camera_id].get_info(rs.camera_info.serial_number))
        
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8,30)
        #self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 10)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.pipeline_profile = self.pipeline.start(self.config)
        self.device = self.pipeline_profile.get_device()
        advanced_mode = rs.rs400_advanced_mode(self.device)
        self.mtx = self.getIntrinsics()
        #with open(r"config/d435_high_accuracy.json", 'r') as file:
        #    json_text = file.read().strip()
        #advanced_mode.load_json(json_text)

        self.hole_filling = rs.hole_filling_filter()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # cam init
        print('cam init ...')
        i = 60
        while i>0:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            i -= 1
        print('cam init done.')

    def get_data(self, hole_filling=False):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if hole_filling:
                depth_frame = self.hole_filling.process(depth_frame)
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            break
        return color_image, depth_image

    def inpaint(self, img, missing_value=0):
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

    def getXYZRGB(self,color, depth, robot_pose, camee_pose, camIntrinsics, inpaint=True):
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
            depth = self.inpaint(depth)
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
        
        # heightIMG, widthIMG, _ = color.shape
        # # heightIMG = 720
        # # widthIMG = 1280
        # depthImg = depth / 1000.
        # # depthImg = depth
        # if inpaint:
        #     depthImg = self.inpaint(depthImg)
        # robot_pose = np.dot(robot_pose, camee_pose)

        # [pixX, pixY] = np.meshgrid(np.arange(widthIMG), np.arange(heightIMG))
        # camX = -(pixX - camIntrinsics[0][2]) * depthImg / camIntrinsics[0][0]
        # camY = (pixY - camIntrinsics[1][2]) * depthImg / camIntrinsics[1][1]
        # camZ = depthImg

        # camPts = [camY.reshape(camY.shape + (1,)), camX.reshape(camX.shape + (1,)), camZ.reshape(camZ.shape + (1,))]
        # camPts = np.concatenate(camPts, 2)
        # camPts = camPts.reshape((camPts.shape[0] * camPts.shape[1], camPts.shape[2]))  # shape = (heightIMG*widthIMG, 3)
        # worldPts = np.dot(robot_pose[:3, :3], camPts.transpose()) + robot_pose[:3, 3].reshape(3,1)  # shape = (3, heightIMG*widthIMG)
        # #worldPts = camPts.T
        # rgb = color.reshape((-1, 3)) / 255.
        # print(rgb[:10])
        # xyzrgb = np.hstack((worldPts.T, rgb))
        # # xyzrgb = self.getleft(xyzrgb)
        # return xyzrgb

    def getleft(self, obj1):
        index = np.bitwise_and(obj1[:, 0] < 1.2, obj1[:, 0] > 0.2)
        index = np.bitwise_and(obj1[:, 1] < 0.5, index)
        index = np.bitwise_and(obj1[:, 1] > -0.5, index)
        # index = np.bitwise_and(obj1[:, 2] > -0.1, index)
        index = np.bitwise_and(obj1[:, 2] > 0.35, index)
        index = np.bitwise_and(obj1[:, 2] < 0.7, index)
        return obj1[index]

    def getIntrinsics(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        print(intrinsics)
        mtx = [intrinsics.width,intrinsics.height,intrinsics.ppx,intrinsics.ppy,intrinsics.fx,intrinsics.fy]
        camIntrinsics = np.array([[mtx[4],0,mtx[2]],
                                  [0,mtx[5],mtx[3]],
                                 [0,0,1.]])
        return camIntrinsics

    
    def __del__(self):
        self.pipeline.stop()


def live_application():
    capture = CameraD400()
    pygame.init()
    display = pygame.display.set_mode((640, 480))
    while True:
        img, dep = capture.get_data()
        display.blit(pygame.surfarray.make_surface(img),(0, 0))
        pygame.display.update()
    
    # capture = RealSenseCapture()
    # pygame.init()
    # display = pygame.display.set_mode((640, 480))

    # while True:
    #     img = capture.read()
    #     display.blit(pygame.surfarray.make_surface(img),(0, 0))
    #     pygame.display.update()

def test():
    capture = CameraD400(1)
    chessboard_size = (5, 7)  # Adjust this value according to your chessboard size
    print(capture.getIntrinsics())
    
    saved_img = 0
    intr = capture.getIntrinsics()
    while True:
        img, depth = capture.get_data()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        cv2.imshow('Image', img)
        cv2.waitKey(100)  # Display the image for 500ms

    cv2.destroyAllWindows()
  
def capture():
    
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../lib_py'))
    import flexivrdk

    robot_ip = "192.168.2.100"
    local_ip = "192.168.2.102"
    cam = CameraD400(0)
    cam_wrist = CameraD400(1)
    chessboard_size = (11, 8)
    
    saved_img = 0
    
    dt = 0.5 # detect every dt seconds
    
    log = flexivrdk.Log()
    robot = flexivrdk.Robot(robot_ip, local_ip)
    robot_states = flexivrdk.RobotStates()
    mode = flexivrdk.Mode
    
    if robot.isFault():
        log.warn("Fault occurred on robot server, trying to clear ...")
        robot.clearFault()
        time.sleep(2)
        # Check again
        if robot.isFault():
            log.error("Fault cannot be cleared, exiting ...")
            return
        log.info("Fault on robot server is cleared")
    
    while True:
        img, _ = cam.get_data()
        img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        img_wrist, _ = cam_wrist.get_data()
        img_wrist = img_wrist.copy()
        gray_ = cv2.cvtColor(img_wrist, cv2.COLOR_BGR2GRAY)    
        ret_, corners_ = cv2.findChessboardCorners(gray_, chessboard_size, None)
        
        if ret and ret_:
            saved_img += 1
            
            robot.getRobotStates(robot_states)
            flange_pose = robot_states.flangePose
            np.savetxt(f"/home/rhos/Desktop/cali_5/{saved_img:03}.txt", flange_pose)
            
            filename = f"/home/rhos/Desktop/cali_5/side/{saved_img:03}.png"
            cv2.imwrite(filename, img)
            
            filename = f"/home/rhos/Desktop/cali_5/wrist/{saved_img:03}.png"
            cv2.imwrite(filename, img_wrist)
            print(f"Saved {filename}")
            
        if ret:
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        if ret_:
            cv2.drawChessboardCorners(img_wrist, chessboard_size, corners_, ret_)
        
        # display img and img_wrist in one window
        img = np.hstack((img, img_wrist))
        cv2.imshow('Image', img)
        cv2.waitKey(1)
        
        
        time.sleep(dt)

    
if __name__ == "__main__":
    # live_application()
    # capture()
    test()
    