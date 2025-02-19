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

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

from hardware.my_device.robot import FlexivRobot, FlexivGripper
from hardware.my_device.camera import CameraD400
from hardware.my_device.keyboard import Keyboard

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
    # overwrite some config values according to evaluation config
    cfg.policy.num_inference_steps = eval_cfg.policy.num_inference_steps
    # if rank == 0:
    #     pathlib.Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    cls = hydra.utils.get_class(cfg._target_)
    # workspace = cls(cfg, rank, world_size, device_id, device)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

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
        print(f"Evaluation episode: {episode_idx}")
        keyboard.finish = False
        time.sleep(5)

        img_0_history = torch.zeros((To, *img_shape), device=device)
        img_1_history = torch.zeros((To, *img_shape), device=device)
        state_history = torch.zeros((To, *state_shape), device=device)
        action_seq = torch.zeros((Ta, 8), device=device)

        last_p = robot.init_pose[:3]
        last_r = R.from_quat(robot.init_pose[3:])

        # Policy inference
        j = 0
        while j < max_episode_length:
            # 'f' to end the episode
            if keyboard.finish:
                robot.send_tcp_pose(robot.init_pose)
                time.sleep(1.5)
                gripper.move(gripper.max_width)
                time.sleep(0.5)
                print("Reset!")
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
                'side_img': img_0_history.unsqueeze(0),
                'wrist_img': img_1_history.unsqueeze(0),
                'qpos': state_history.unsqueeze(0)
            }
            # Predict qpos actions
            curr_action = policy.predict_action(curr_obs)
            np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
            action_seq = np_action_dict['action'][0]

            for step in range(Ta):
                if step > 0:
                    start_time = time.time()
                # target_pos = robot.init_pose[:3] + action_seq[step, :3]
                # target_rot = R.from_quat(robot.init_pose[3:]) * R.from_quat(action_seq[step, 3:7])
                # robot.send_tcp_pose(np.concatenate((target_pos, target_rot.as_quat()), 0))
                # robot.send_tcp_pose(action_seq[step, :7])
                curr_p = last_p + action_seq[step, :3]
                curr_r = last_r * R.from_quat(action_seq[step, 3:7])
                robot.send_tcp_pose(np.concatenate((curr_p, curr_r.as_quat()), 0))
                gripper.move(action_seq[step, 7])
                last_p = curr_p
                last_r = curr_r
                time.sleep(max(1 / fps - (time.time() - start_time), 0))
                j += 1

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