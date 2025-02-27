import pathlib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import torch
import numpy as np
import dill
import time
from scipy.spatial.transform import Rotation as R
import cv2
from torchvision.transforms import Compose, Resize
from torchvision.transforms import InterpolationMode
import pygame

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.pytorch_util import dict_apply

from hardware.my_device.robot import FlexivRobot, FlexivGripper
from hardware.my_device.camera import CameraD400
from hardware.my_device.keyboard import Keyboard
from hardware.my_device.sigma import Sigma7
from hardware.my_device.logitechG29_wheel import Controller

camera_serial = ["038522063145", "104422070044"]

def main(rank, eval_cfg, device_ids):
    fps = 10  # TODO
    world_size = len(device_ids)
    device_id = device_ids[rank]
    device = f"cuda:{device_id}"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # load checkpoint
    payload = torch.load(open(eval_cfg.checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    rel_ee_pose = cfg.task.dataset.rel_ee_pose # Determines the action space

    # overwrite some config values according to evaluation config
    cfg.policy.num_inference_steps = eval_cfg.policy.num_inference_steps
    cfg.output_dir = eval_cfg.output_dir
    # if rank == 0:
    #     pathlib.Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, rank, world_size, device_id, device)
    # workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model.module
    if cfg.training.use_ema:
        policy = workspace.ema_model.module

    policy.to(device)
    policy.eval()

    # Extract some hyperparameters from the config
    To = policy.n_obs_steps
    Ta = policy.n_action_steps
    img_shape = cfg.task['shape_meta']['obs']['wrist_img']['shape']
    state_shape = cfg.task['shape_meta']['obs']['qpos']['shape']

    BICUBIC = InterpolationMode.BICUBIC
    image_processor = Compose([Resize(img_shape[1:], interpolation=BICUBIC)])

    # Overwritten by evaluation config specifically
    Ta = eval_cfg.Ta

    # Initialize hardware
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    cameras = [CameraD400(s) for s in camera_serial]
    keyboard = Keyboard()
    sigma = Sigma7()
    pygame.init()
    controller = Controller(0)

    # Initialize demonstration buffer
    zarr_path = os.path.join(eval_cfg.save_buffer_path, 'replay_buffer.zarr')
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='a')

    # Evaluation starts here
    max_episode_length = 400
    episode_list = [x for x in range(eval_cfg.num_episode) if (x + 1) % world_size == rank]

    for episode_idx in range(eval_cfg.num_episode):
        if keyboard.quit:
            break
        if episode_idx not in episode_list:
            continue
        print(f"Rollout episode: {episode_idx}")

        # Reset keyboard states
        keyboard.finish = False
        keyboard.help = False
        keyboard.infer = False
        keyboard.discard = False
        time.sleep(1)

        # Initialize episode buffers
        tcp_pose = []
        joint_pos = []
        action = []
        wrist_cam = []
        side_cam = []

        # Reset the robot to home pose
        robot.send_tcp_pose(robot.init_pose)
        time.sleep(2)
        gripper.move(gripper.max_width)
        time.sleep(0.5)
        print("Reset!")
        # Reset the sigma pose as well
        sigma.reset()

        # Initialize obs history buffer
        policy_img_0_history = torch.zeros((To, *img_shape), device=device)
        policy_img_1_history = torch.zeros((To, *img_shape), device=device)
        policy_state_history = torch.zeros((To, *state_shape), device=device)

        _, state, _, _ = robot.get_robot_state()
        state = torch.from_numpy(np.array(state))
        cam_data = []
        for camera in cameras:
            color_image, _ = camera.get_data()
            cam_data.append(color_image)
        original_img_0 = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        original_img_1 = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        policy_img_0 = original_img_0 / 255.
        policy_img_1 = original_img_1 / 255.

        for idx in range(policy_state_history.shape[0]):
            policy_img_0_history[idx] = policy_img_0
            policy_img_1_history[idx] = policy_img_1
            policy_state_history[idx] = state

        # Keep track of pose from the last frame for relative action space
        last_p = robot.init_pose[:3]
        last_r = R.from_quat(robot.init_pose[3:][[1,2,3,0]])
        # Keep track of throttle usage for human intervention (Default to True because the teleop should follow up from arbitrary pose)
        last_throttle = True
        sigma.detach()
        detach_tcp, _, _, _ = robot.get_robot_state()
        detach_pos = np.array(detach_tcp[:3])
        detach_rot = R.from_quat(np.array(detach_tcp[3:])[[1,2,3,0]])
        j = 0 # Episode timestep

        while True:
            # ===========================================================
            # Policy inference loop
            # ===========================================================
            print("=========== Policy inference ============")
            while not keyboard.finish and not keyboard.discard and not keyboard.help:
                if j >= max_episode_length:
                    break

                start_time = time.time()
                _, state, _, _ = robot.get_robot_state()
                state = torch.from_numpy(np.array(state))
                cam_data = []
                for camera in cameras:
                    color_image, _ = camera.get_data()
                    cam_data.append(color_image)
                img_0 = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) / 255.
                img_1 = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) / 255.

                # Update the observation history
                policy_img_0_history[:-1] = policy_img_0_history[1:]
                policy_img_0_history[-1] = img_0
                policy_img_1_history[:-1] = policy_img_1_history[1:]
                policy_img_1_history[-1] = img_1
                policy_state_history[:-1] = policy_state_history[1:]
                policy_state_history[-1] = state
                curr_obs = {
                    'side_img': policy_img_0_history.unsqueeze(0),
                    'wrist_img': policy_img_1_history.unsqueeze(0),
                    'qpos': policy_state_history.unsqueeze(0)
                }

                # Predict qpos actions
                curr_action = policy.predict_action(curr_obs)
                np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
                action_seq = np_action_dict['action'][0]

                for step in range(Ta):
                    if step > 0:
                        start_time = time.time()
                    if not rel_ee_pose:
                        curr_p = action_seq[step, :3]
                        curr_r = R.from_quat(action_seq[step, [4,5,6,3]])
                    else:
                        curr_p = last_p + action_seq[step, :3] @ last_r.as_matrix().T
                        curr_r = last_r * R.from_quat(action_seq[step, [4,5,6,3]])

                    # Record demo data
                    cam_data = []
                    for camera in cameras:
                        color_image, depth_image = camera.get_data()
                        cam_data.append((color_image, depth_image))
                    tcpPose, jointPose, _, _ = robot.get_robot_state()
                    demo_gripper_action = 1 if action_seq[step, 7] >= 0.5 else 0

                    robot.send_tcp_pose(np.concatenate((curr_p, curr_r.as_quat()[[3,0,1,2]]), 0))
                    target_width = gripper.max_width if action_seq[step, 7] < 0.5 else 0 # Threshold could be adjusted at inference time
                    gripper.move(target_width)

                    # Save demonstrations to the buffer
                    wrist_image = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1][0].copy(), cv2.COLOR_BGR2RGB)).\
                                      permute(2,0,1)).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                    side_image = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0][0].copy(), cv2.COLOR_BGR2RGB)).\
                                                permute(2,0,1)).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                    wrist_cam.append(wrist_image)
                    side_cam.append(side_image)
                    tcp_pose.append(tcpPose)
                    joint_pos.append(jointPose)
                    action.append(np.concatenate((curr_p, curr_r.as_quat()[[3,0,1,2]], [demo_gripper_action])))

                    time.sleep(max(1 / fps - (time.time() - start_time), 0))
                    j += 1

                last_p = curr_p
                last_r = curr_r

            # Reset the signal of request for help 
            keyboard.help = False
            # Compensate for the transformation of robot tcp pose during sigma detachment
            resume_tcp, _, _, _ = robot.get_robot_state()
            resume_pos = np.array(resume_tcp[:3])
            resume_rot = R.from_quat(np.array(resume_tcp[3:])[[1,2,3,0]])
            translate = resume_pos - detach_pos
            rotation = detach_rot.inv() * resume_rot
            sigma.transform_from_robot(translate, rotation)

            print("============ Human intervention =============")
            # ============================================================
            # Human intervention loop
            # ============================================================
            while not keyboard.finish and not keyboard.discard and not keyboard.infer:
                if j >= max_episode_length:
                    break
                
                start_time = time.time()
                cam_data = []
                for camera in cameras:
                    color_image, depth_image = camera.get_data()
                    cam_data.append((color_image, depth_image))
                tcpPose, jointPose, _, _ = robot.get_robot_state()
                
                diff_p, diff_r, width = sigma.get_control()
                diff_p = robot.init_pose[:3] + diff_p
                diff_r = R.from_quat(robot.init_pose[3:][[1,2,3,0]]) * diff_r

                # Get throttle pedal state
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        keyboard.quit = True
                throttle = controller.get_throttle()
                # If throttle is activated, freeze the robot while adjusting the teleop device.
                if throttle < -0.9:
                    if not last_throttle:
                        sigma.detach()
                        last_throttle = True
                    continue
                if last_throttle:
                    last_throttle = False
                    sigma.resume()
                    continue

                # Send command.
                robot.send_tcp_pose(np.concatenate((diff_p, diff_r.as_quat()[[3,0,1,2]]), 0))
                gripper.move_from_sigma(width)
                gripper_action = 1 if width < 500 else 0

                # Renew policy obs buffer for the next inference
                img_0 = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0][0].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) / 255.
                img_1 = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1][0].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) / 255.
                policy_img_0_history[:-1] = policy_img_0_history[1:]
                policy_img_0_history[-1] = img_0
                policy_img_1_history[:-1] = policy_img_1_history[1:]
                policy_img_1_history[-1] = img_1
                policy_state_history[:-1] = policy_state_history[1:]
                policy_state_history[-1] = torch.from_numpy(np.array(jointPose))
                
                # Save demonstrations to the buffer
                wrist_image = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1][0].copy(), cv2.COLOR_BGR2RGB)).\
                                      permute(2,0,1)).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                side_image = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0][0].copy(), cv2.COLOR_BGR2RGB)).\
                                            permute(2,0,1)).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                wrist_cam.append(wrist_image)
                side_cam.append(side_image)
                tcp_pose.append(tcpPose)
                joint_pos.append(jointPose)
                action.append(np.concatenate((diff_p, diff_r.as_quat()[[3,0,1,2]], [gripper_action])))

                time.sleep(max(1 / fps - (time.time() - start_time), 0))
                j += 1

            # Reset the signal of request for inference
            keyboard.infer = False
            # Detach the teleop device except when human intervention
            sigma.detach()
            last_throttle = True
            detach_tcp, _, _, _ = robot.get_robot_state()
            detach_pos = np.array(detach_tcp[:3])
            detach_rot = R.from_quat(np.array(detach_tcp[3:])[[1,2,3,0]])

            # This episode fails to accomplish the task
            if j >= max_episode_length or keyboard.discard:
                break

            # Save the demonstrations to the replay buffer
            if keyboard.finish:
                episode = dict()
                episode['wrist_cam'] = np.stack(wrist_cam, axis=0)
                episode['side_cam'] = np.stack(side_cam, axis=0)
                episode['tcp_pose'] = np.stack(tcp_pose, axis=0)
                episode['joint_pos'] = np.stack(joint_pos, axis=0)
                episode['action'] = np.stack(action, axis=0)
                replay_buffer.add_episode(episode, compressors='disk')
                episode_id = replay_buffer.n_episodes - 1
                print('Saved episode ', episode_id)
                break

        robot.send_tcp_pose(robot.init_pose)
        time.sleep(2)
        gripper.move(gripper.max_width)
        time.sleep(0.5)
        print("Reset!")

        # For task configuration reset
        time.sleep(5)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    with hydra.initialize(config_path='diffusion_policy/config'):
        cfg = hydra.compose(config_name=pathlib.Path(__file__).stem)

    device_ids = [int(x) for x in cfg.device_ids.split(",")]
    os.environ["MASTER_ADDR"] = "localhost"
    port = 29999
    import socket
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                print(port)
                break
        except OSError:
            port += 1
    os.environ["MASTER_PORT"] = f"{port}"
    if len(device_ids) == 1:
        main(0, cfg, device_ids)
    elif len(device_ids) > 1:
        torch.multiprocessing.spawn(main, args=(cfg, device_ids), nprocs=len(device_ids), join=True)