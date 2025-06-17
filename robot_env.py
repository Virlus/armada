import numpy as np
import torch
import time
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R
import pygame
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms import InterpolationMode

from hardware.my_device.robot import FlexivRobot, FlexivGripper
from hardware.my_device.camera import CameraD400
from hardware.my_device.keyboard import Keyboard
from hardware.my_device.sigma import Sigma7
from hardware.my_device.logitechG29_wheel import Controller

# Sirius-specific macros
HUMAN = 0
ROBOT = 1
PRE_INTV = 2
INTV = 3
INTV_STEPS = 15

class RobotEnv:
    def __init__(self, camera_serial=None, img_shape=None, fps=10):
        if camera_serial is None:
            camera_serial = ["135122075425", "135122070361"]
        self.camera_serial = camera_serial
        self.fps = fps
        self.img_shape = img_shape
        
        # Initialize hardware components
        self.robot = FlexivRobot()
        self.gripper = FlexivGripper(self.robot)
        self.cameras = [CameraD400(s) for s in self.camera_serial]
        self.keyboard = Keyboard()
        self.sigma = Sigma7()
        pygame.init()
        self.controller = Controller(0)
        
        # Setup image processors
        BICUBIC = InterpolationMode.BICUBIC
        self.side_image_processor = Compose([
            Resize((img_shape[1]+8, img_shape[2]+8), interpolation=BICUBIC),
            CenterCrop((img_shape[1], img_shape[2]))
        ])
        self.wrist_image_processor = Resize((img_shape[1], img_shape[2]), interpolation=BICUBIC)
        
        # Keep track of throttle usage for human intervention
        self.last_throttle = True
        
    def reset_robot(self, random_init=False, random_init_pose=None):
        if random_init and random_init_pose is not None:
            self.robot.send_tcp_pose(random_init_pose)
        else:
            self.robot.send_tcp_pose(self.robot.init_pose)
        time.sleep(2)
        self.gripper.move(self.gripper.max_width)
        time.sleep(0.5)
        print("Reset!")
        
        # Reset the sigma pose as well
        self.sigma.reset()
        if random_init and random_init_pose is not None:
            random_p_drift = random_init_pose[:3] - self.robot.init_pose[:3]
            random_r_drift = R.from_quat(self.robot.init_pose[3:7], scalar_first=True).inv() * R.from_quat(random_init_pose[3:7], scalar_first=True)
            self.sigma.transform_from_robot(random_p_drift, random_r_drift)
        
        return self.get_robot_state()
    
    def get_robot_state(self):
        """Get robot state, images, and joint positions"""
        # Get robot state
        tcp_pose, joint_pos, _, _ = self.robot.get_robot_state()
        
        # Get camera images
        cam_data = []
        for camera in self.cameras:
            color_image, _ = camera.get_data()
            cam_data.append(color_image)
            
        # Process images
        side_img = self.side_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        wrist_img = self.wrist_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        
        return {
            'tcp_pose': tcp_pose,
            'joint_pos': joint_pos,
            'side_img': side_img,
            'wrist_img': wrist_img,
            'side_img_raw': cam_data[0].copy(),
            'wrist_img_raw': cam_data[1].copy()
        }
    
    def deploy_action(self, tcp_action, gripper_action):
        self.robot.send_tcp_pose(tcp_action)
        self.gripper.move(gripper_action)
    
    def save_scene_images(self, output_dir, episode_idx):
        """Save scene images to output directory"""
        cam_data = []
        for camera in self.cameras:
            color_image, _ = camera.get_data()
            cam_data.append(color_image)
        side_img = cv2.cvtColor(cam_data[0].copy(), cv2.COLOR_BGR2RGB)
        wrist_img = cv2.cvtColor(cam_data[1].copy(), cv2.COLOR_BGR2RGB)
        Image.fromarray(side_img).save(f"{output_dir}/side_{episode_idx}.png")
        Image.fromarray(wrist_img).save(f"{output_dir}/wrist_{episode_idx}.png")
        return side_img, wrist_img
    
    def align_scene(self, output_dir, episode_idx):
        """Align current scene with reference images"""
        ref_side_img = cv2.imread(f"{output_dir}/side_{episode_idx}.png")
        ref_wrist_img = cv2.imread(f"{output_dir}/wrist_{episode_idx}.png")
        cv2.namedWindow("Side", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Wrist", cv2.WINDOW_AUTOSIZE)
        
        while not self.keyboard.ctn:
            cam_data = []
            for camera in self.cameras:
                color_image, _ = camera.get_data()
                cam_data.append(color_image)
            side_img = cam_data[0].copy()
            wrist_img = cam_data[1].copy()
            cv2.imshow("Side", (np.array(side_img) * 0.5 + np.array(ref_side_img) * 0.5).astype(np.uint8))
            cv2.imshow("Wrist", (np.array(wrist_img) * 0.5 + np.array(ref_wrist_img) * 0.5).astype(np.uint8))
            cv2.waitKey(1)
        
        self.keyboard.ctn = False
        cv2.destroyAllWindows()
    
    def detach_sigma(self):
        """Detach sigma device and store TCP pose"""
        self.sigma.detach()
        detach_tcp, _, _, _ = self.robot.get_robot_state()
        detach_pos = np.array(detach_tcp[:3])
        detach_rot = R.from_quat(np.array(detach_tcp[3:]), scalar_first=True)
        return detach_pos, detach_rot
    
    def human_teleop_step(self, last_p, last_r):
        """Execute one step of human teleoperation"""
        start_time = time.time()
        
        # Get camera data and robot state
        state_data = self.get_robot_state()
        tcp_pose = state_data['tcp_pose']
        joint_pos = state_data['joint_pos']
        
        # Get teleop controls
        diff_p, diff_r, width = self.sigma.get_control()
        diff_p = self.robot.init_pose[:3] + diff_p
        diff_r = R.from_quat(self.robot.init_pose[3:7], scalar_first=True) * diff_r
        curr_p_action = diff_p - last_p
        curr_r_action = last_r.inv() * diff_r
        last_p = diff_p
        last_r = diff_r
        
        # Check throttle pedal state (for teleop pausing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.keyboard.quit = True
        
        throttle = self.controller.get_throttle()
        if throttle < -0.9:
            if not self.last_throttle:
                self.sigma.detach()
                self.last_throttle = True
            return None, last_p, last_r
        
        if self.last_throttle:
            self.last_throttle = False
            self.sigma.resume()
            last_p, last_r, _ = self.sigma.get_control()
            last_p = last_p + self.robot.init_pose[:3]
            last_r = R.from_quat(self.robot.init_pose[3:7], scalar_first=True) * last_r
            return None, last_p, last_r
        
        # Send command to robot
        self.robot.send_tcp_pose(np.concatenate((diff_p, diff_r.as_quat(scalar_first=True)), 0))
        self.gripper.move_from_sigma(width)
        gripper_action = self.gripper.max_width * width / 1000
        
        # Save demo data for return
        processed_data = {
            'wrist_img': state_data['wrist_img'],
            'side_img': state_data['side_img'],
            'tcp_pose': tcp_pose,
            'joint_pos': joint_pos,
            'action': np.concatenate((curr_p_action, curr_r_action.as_quat(scalar_first=True), [gripper_action])),
            'action_mode': INTV
        }
        
        # Sleep to maintain fps
        time.sleep(max(1 / self.fps - (time.time() - start_time), 0))
        
        return processed_data, diff_p, diff_r
    
    def rewind_robot(self, curr_pos, curr_rot, p_action, r_action, gripper_action):
        """Rewind the robot by applying inverse actions"""
        
        # Apply inverse action
        curr_pos = curr_pos - p_action
        curr_rot = curr_rot * R.from_quat(r_action, scalar_first=True).inv()
        
        # Send command
        self.robot.send_tcp_pose(np.concatenate((curr_pos, curr_rot.as_quat(scalar_first=True)), 0))
        self.gripper.move(gripper_action)
        
        return curr_pos, curr_rot
    
    def cleanup(self):
        """Clean up resources"""
        for camera in self.cameras:
            camera.release()
        self.sigma.cleanup()
        pygame.quit()


def postprocess_action_mode(action_mode: np.ndarray):
    """Postprocessing for action mode identification"""
    for i in range(1, len(action_mode)):
        if action_mode[i] == INTV and action_mode[i-1] == ROBOT:
            pre_intv_indices = np.arange(max(0, i-INTV_STEPS), i)
            pre_intv_indices = pre_intv_indices[np.where(action_mode[pre_intv_indices] == INTV, False, True)]
            action_mode[pre_intv_indices] = PRE_INTV
    return action_mode 