import pathlib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diffusion_policy"))
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
from PIL import Image

from diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer

from hardware.robot_env import RobotEnv
from hardware.my_device.macros import CAM_SERIAL
from util.episode_utils import EpisodeManager

def main(rank, eval_cfg, device_ids):
    fps = 10  # TODO
    world_size = len(device_ids)
    device_id = device_ids[rank]
    device = f"cuda:{device_id}"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # load checkpoint
    payload = torch.load(open(eval_cfg.checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    rel_ee_pose = True # Hacked because currently the action space is always relative

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
    if 'obs_encoder' in cfg.policy:
        cfg.policy.obs_encoder.pretrained_path = None

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

    # Overwritten by evaluation config specifically
    seed = eval_cfg.seed
    np.random.seed(seed)
    Ta = eval_cfg.Ta
    save_img = False
    output_dir = os.path.join(eval_cfg.output_dir, f"seed_{seed}")
    if os.path.isdir(output_dir):
        print(f"Output directory {output_dir} already exists, will not overwrite it.")
    else:
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        save_img = True

    # Initialize robot environment
    robot_env = RobotEnv(camera_serial=CAM_SERIAL, img_shape=img_shape, fps=fps)
    
    # Initialize EpisodeManager
    episode_manager = EpisodeManager(
        policy=policy,
        obs_rot_transformer=obs_rot_transformer,
        action_rot_transformer=action_rot_transformer,
        obs_feature_dim=ee_pose_dim,
        img_shape=img_shape,
        state_type=state_type,
        state_shape=state_shape,
        action_dim=action_dim,
        To=To,
        Ta=Ta,
        device=device,
        num_samples=1  # Single sample for evaluation
    )

    max_episode_length = 400
    episode_list = [x for x in range(eval_cfg.num_episode) if (x + 1) % world_size == rank]

    for episode_idx in range(eval_cfg.num_episode):
        if episode_idx not in episode_list:
            continue
        print(f"Evaluation episode: {episode_idx}")
        robot_env.keyboard.finish = False
        time.sleep(5)

        # Reset robot environment
        if eval_cfg.random_init:
            random_init_pose = robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
            robot_env.reset_robot(random_init=True, random_init_pose=random_init_pose)
        else:
            robot_env.reset_robot()

        # Reset episode manager and initialize observation history
        episode_manager.reset_observation_history()

        # Get initial state using RobotEnv
        robot_state = robot_env.get_robot_state()
        if 'ee_pose' in cfg.shape_meta['obs']:
            initial_state = robot_state['tcp_pose']
        else:
            initial_state = robot_state['joint_pos']

        # Use processed images from RobotEnv
        policy_img_0 = robot_state['policy_side_img'] / 255.
        policy_img_1 = robot_state['policy_wrist_img'] / 255.

        # Initialize observation history using EpisodeManager
        for idx in range(To):
            episode_manager.update_observation(policy_img_0, policy_img_1, initial_state)

        # Initialize pose tracking in EpisodeManager
        if eval_cfg.random_init:
            episode_manager.initialize_pose(random_init_pose[:3], random_init_pose[3:7])
        else:
            episode_manager.initialize_pose(robot_env.robot.init_pose[:3], robot_env.robot.init_pose[3:7])

        # Handle scene alignment and image saving
        if save_img or not os.path.isfile(os.path.join(output_dir, f"side_{episode_idx}.png")):
            robot_env.save_scene_images(output_dir, episode_idx)
        else:
            robot_env.align_scene_with_file(output_dir, episode_idx)
        
        # Policy inference
        j = 0
        while j < max_episode_length:
            # 'f' to end the episode
            if robot_env.keyboard.finish:
                robot_env.reset_robot()
                print("Reset!")
                break
            
            start_time = time.time()
            
            # Get state using RobotEnv
            robot_state = robot_env.get_robot_state()
            if 'ee_pose' in cfg.shape_meta['obs']:
                state = robot_state['tcp_pose']
            else:
                state = robot_state['joint_pos']
            
            # Use processed images from RobotEnv
            img_0 = robot_state['policy_side_img'] / 255.
            img_1 = robot_state['policy_wrist_img'] / 255.

            # Update observation history using EpisodeManager
            episode_manager.update_observation(img_0, img_1, state)

            # Get policy observation and predict actions
            curr_obs = episode_manager.get_policy_observation()
            curr_action = policy.predict_action(curr_obs)
            np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
            action_seq = np_action_dict['action'][0][:Ta]

            # Derive the action chunk
            if not rel_ee_pose:
                p_chunk = action_seq[:, :3]
                quat_chunk = action_rot_transformer.inverse(action_seq[:, 3:action_dim-1])
                r_chunk = R.from_quat(quat_chunk, scalar_first=True)

            for step in range(Ta):
                if step > 0:
                    start_time = time.time()
                
                # Action calculation
                if rel_ee_pose:
                    # Get absolute action using EpisodeManager
                    deployed_action, gripper_action, curr_p, curr_r, curr_p_action, curr_r_action = episode_manager.get_absolute_action_for_step(
                        action_seq[np.newaxis, :], step
                    )
                else:
                    curr_p = p_chunk[step]
                    curr_r = r_chunk[step]
                    deployed_action = np.concatenate((curr_p, curr_r.as_quat(scalar_first=True)), 0)
                    gripper_action = [action_seq[step, -1]]

                # Deploy action using RobotEnv
                robot_env.deploy_action(deployed_action, gripper_action[0])
                
                time.sleep(max(1 / fps - (time.time() - start_time), 0))
                j += 1

        if j == max_episode_length:
            robot_env.reset_robot()
            print("Reset!")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    with hydra.initialize(config_path='diffusion_policy/diffusion_policy/config'):
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