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
import torch.nn.functional as F

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from hardware.my_device.robot import FlexivRobot, FlexivGripper
from hardware.my_device.camera import CameraD400
from hardware.my_device.keyboard import Keyboard

camera_serial = ["038522063145", "104422070044"]
base2cam_T_path = "/home/yuwenye/projects/human-in-the-loop/hardware/calib/workspace/flexiv/calib/out/cali_side/camT.npy"
intrinsics_path = "/home/yuwenye/projects/human-in-the-loop/hardware/calib/workspace/flexiv/calib/out/cali_side/intrinsic.npy"

def main(rank, eval_cfg, device_ids):
    fps = 10  # TODO
    world_size = len(device_ids)
    device_id = device_ids[rank]
    device = f"cuda:{device_id}"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # load extrinsic matrix and intrinsic matrix
    base2cam_T = np.load(base2cam_T_path)
    cam2base_T = np.linalg.inv(base2cam_T)
    intrinsic = np.load(intrinsics_path)

    # To visualize action distribution
    num_action_samples = eval_cfg.num_action_samples

    # load checkpoint
    payload = torch.load(open(eval_cfg.checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    # rel_ee_pose = cfg.task.dataset.rel_ee_pose # Determines the action space
    rel_ee_pose = True # Hacked because currently the action space is always relative
    # cfg.shape_meta = eval_cfg.shape_meta # Hacked for the same reason as above

    # rotation transformation for action space and observation space
    action_dim = cfg.shape_meta['action']['shape'][0]
    action_rot_transformer = None
    obs_rot_transformer = None
    # Check if there's need for transforming rotation representation
    if 'rotation_rep' in cfg.shape_meta['action']:
        action_rot_transformer = RotationTransformer(from_rep='quaternion', to_rep=cfg.shape_meta['action']['rotation_rep'])
    if 'ee_pose' in cfg.shape_meta['obs']:
        ee_pose_dim = cfg.shape_meta['obs']['ee_pose']['shape'][0]
        state_type = 'ee_pose'
        if 'rotation_rep' in cfg.shape_meta['obs']['ee_pose']:
            obs_rot_transformer = RotationTransformer(from_rep='quaternion', to_rep=cfg.shape_meta['obs']['ee_pose']['rotation_rep'])
    else:
        ee_pose_dim = cfg.shape_meta['obs']['qpos']['shape'][0]
        state_type = 'qpos'

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
    if 'ee_pose' in cfg.shape_meta['obs']:
        state_shape = cfg.task['shape_meta']['obs']['ee_pose']['shape']
    else:
        state_shape = cfg.task['shape_meta']['obs']['qpos']['shape']

    BICUBIC = InterpolationMode.BICUBIC
    image_processor = Compose([Resize(img_shape[1:], interpolation=BICUBIC)])

    # Overwritten by evaluation config specifically
    seed = int(time.time())
    np.random.seed(seed)
    Ta = eval_cfg.Ta

    # run evaluation
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    cameras = [CameraD400(s) for s in camera_serial]
    keyboard = Keyboard()
    max_episode_length = 400
    episode_list = [x for x in range(eval_cfg.num_episode) if (x + 1) % world_size == rank]

    for episode_idx in range(eval_cfg.num_episode):
        if episode_idx not in episode_list:
            continue
        os.makedirs(f"{eval_cfg.output_dir}/episode_{episode_idx}", exist_ok=True)
        print(f"Evaluation episode: {episode_idx}")
        keyboard.finish = False
        time.sleep(5)

        img_0_history = torch.zeros((To, *img_shape), device=device)
        img_1_history = torch.zeros((To, *img_shape), device=device)
        state_history = torch.zeros((To, *state_shape), device=device)
        action_samples = np.zeros((num_action_samples, Ta, action_dim))

        if eval_cfg.random_init:
            random_init_pose = robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
            robot.send_tcp_pose(random_init_pose)
            time.sleep(2)

        if eval_cfg.random_init:
            last_p = random_init_pose[np.newaxis, :3].repeat(num_action_samples, axis=0)
            last_r = R.from_quat(random_init_pose[np.newaxis, 3:7].repeat(num_action_samples, axis=0), scalar_first=True)
        else:
            last_p = robot.init_pose[np.newaxis, :3].repeat(num_action_samples, axis=0)
            last_r = R.from_quat(robot.init_pose[np.newaxis, 3:7].repeat(num_action_samples, axis=0), scalar_first=True)

        # Policy inference
        j = 0
        while j < max_episode_length:
            # 'f' to end the episode
            if keyboard.finish:
                robot.send_joint_pose(robot.home_joint_pos)
                time.sleep(1.5)
                robot.send_tcp_pose(robot.init_pose)
                time.sleep(1.5)
                gripper.move(gripper.max_width)
                time.sleep(0.5)
                print("Reset!")
                break
            start_time = time.time()
            # _, state, _, _ = robot.get_robot_state()
            if 'ee_pose' in cfg.shape_meta['obs']:
                state = robot.get_robot_state()[0]
            else:
                state = robot.get_robot_state()[1]
            state = np.array(state)

            if obs_rot_transformer is not None:
                tmp_state = np.zeros((ee_pose_dim,))
                tmp_state[:3] = state[:3]
                tmp_state[3:] = obs_rot_transformer.forward(state[3:])
                state = tmp_state
            state = torch.from_numpy(state)
            cam_data = []
            for camera in cameras:
                color_image, _ = camera.get_data()
                cam_data.append(color_image)
            img_0 = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) / 255.
            img_1 = image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) / 255.
            if j == 0:
                for idx in range(state_history.shape[0]):
                    img_0_history[idx] = img_0
                    img_1_history[idx] = img_1
                    state_history[idx] = state
            else:
                # Update the observation history
                img_0_history[:-1] = img_0_history[1:]
                img_0_history[-1] = img_0
                img_1_history[:-1] = img_1_history[1:]
                img_1_history[-1] = img_1
                state_history[:-1] = state_history[1:]
                state_history[-1] = state
            curr_obs = {
                'side_img': img_0_history.unsqueeze(0).repeat(num_action_samples, 1, 1, 1, 1),
                'wrist_img': img_1_history.unsqueeze(0).repeat(num_action_samples, 1, 1, 1, 1),
                state_type: state_history.unsqueeze(0).repeat(num_action_samples, 1, 1)
            }

            # Fetch current RGB observation
            side_vis = cam_data[0].copy()

            # Predict qpos actions
            # for i in range(eval_cfg.num_action_samples):
            curr_action = policy.predict_action(curr_obs)
            np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
            # action_seq = np_action_dict['action'][0, :Ta]
            action_samples = np_action_dict['action'][:, :Ta]
            # action_samples[i] = action_seq

            for step in range(Ta):
                if step > 0:
                    start_time = time.time()
                # Action calculation
                if rel_ee_pose:
                    curr_p = last_p + action_samples[:, step, :3]
                    curr_quat = action_rot_transformer.inverse(action_samples[:, step, 3:action_dim-1])
                    action_rot = R.from_quat(curr_quat, scalar_first=True)
                    curr_r = last_r * action_rot
                    last_p = curr_p
                    last_r = curr_r

                # Annotate on the side image
                cam_R,_ = cv2.Rodrigues(cam2base_T[:3, :3])
                cam_T = cam2base_T[:3, 3] * 1000
                projected_pos = cv2.projectPoints(curr_p * 1000, cam_R, cam_T, cameraMatrix=intrinsic, distCoeffs=None)
                projected_pos = projected_pos[0].squeeze()
                projected_pos = projected_pos.astype(int)
                for i in range(num_action_samples):
                    cv2.circle(side_vis, tuple(projected_pos[i]), 3, (int(255 * i / num_action_samples), 0, 255 - int(255 * i / num_action_samples)), -1)
                # cv2.circle(side_vis, tuple(projected_pos), 3, (0, 0, 255), -1)

                robot.send_tcp_pose(np.concatenate((curr_p[0], curr_r[0].as_quat(scalar_first=True)), 0))
                # target_width = gripper.max_width if action_seq[step, -1] < 0.5 else 0 # Threshold could be adjusted at inference time
                target_width = action_samples[0, step, -1]
                gripper.move(target_width)
                time.sleep(max(1 / fps - (time.time() - start_time), 0))
                j += 1

            last_p = np.expand_dims(last_p[0], axis=0).repeat(num_action_samples, axis=0)
            last_r = R.from_quat(np.expand_dims(last_r[0].as_quat(scalar_first=True), axis=0).repeat(num_action_samples, axis=0), scalar_first=True)

            # Save the annotated image
            cv2.imwrite(f"{eval_cfg.output_dir}/episode_{episode_idx}/step_{j}.png", side_vis)


        if j == max_episode_length:
            robot.send_joint_pose(robot.home_joint_pos)
            time.sleep(1.5)
            robot.send_tcp_pose(robot.init_pose)
            time.sleep(1.5)
            gripper.move(gripper.max_width)
            time.sleep(0.5)
            print("Reset!")

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
        # torch.multiprocessing.spawn(main, args=(cfg, device_ids), nprocs=len(device_ids), join=True)
    elif len(device_ids) > 1:
        torch.multiprocessing.spawn(main, args=(cfg, device_ids), nprocs=len(device_ids), join=True)