import pathlib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diffusion_policy"))
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import re
import hydra
import torch
import numpy as np
import dill
import time
from scipy.spatial.transform import Rotation as R

from diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer

from robot_env import RobotEnv, HUMAN, ROBOT, PRE_INTV, INTV, postprocess_action_mode
from hardware.my_device.macros import CAM_SERIAL
from util.episode_utils import EpisodeManager


def main(rank, eval_cfg, device_ids):
    fps = 10  # TODO
    world_size = len(device_ids)
    device_id = device_ids[rank]
    device = f"cuda:{device_id}"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # Random seed 
    seed = int(time.time())
    np.random.seed(seed)

    # load checkpoint
    payload = torch.load(open(eval_cfg.checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

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
    
    # Inspect the current round (Sirius-specific)
    match_round = re.search(r'round(\d)', eval_cfg.save_buffer_path)
    assert match_round
    num_round = int(match_round.group(1))
    
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
        num_samples=1  # Single sample for rollout
    )

    # Initialize demonstration buffer
    zarr_path = os.path.join(eval_cfg.train_dataset_path, 'replay_buffer.zarr')
    dataset_keys = ['wrist_cam', 'side_cam', 'joint_pos', 'action', 'tcp_pose']
    if 'round' in eval_cfg.train_dataset_path:
        dataset_keys.append('action_mode')
        replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=dataset_keys)
    else:
        replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=dataset_keys)
        replay_buffer.data['action_mode'] = np.full((replay_buffer.n_steps, ), HUMAN)

    # Evaluation starts here
    episode_idx = 0
    max_episode_length = 600

    try:
        while True: 
            if robot_env.keyboard.quit:
                break
            print(f"Rollout episode: {episode_idx}")

            # Reset keyboard states
            robot_env.keyboard.finish = False
            robot_env.keyboard.help = False
            robot_env.keyboard.infer = False
            robot_env.keyboard.discard = False
            time.sleep(1)

            # Initialize episode buffers
            tcp_pose = []
            joint_pos = []
            action = []
            action_mode = []
            wrist_cam = []
            side_cam = []

            # Reset the robot using RobotEnv
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
            
            # Strictly align with previous scenes using RobotEnv
            if save_img or not os.path.isfile(os.path.join(output_dir, f"side_{episode_idx}.png")):
                robot_env.save_scene_images(output_dir, episode_idx)
            else:
                robot_env.align_scene_with_file(output_dir, episode_idx)

            # Detach sigma and get initial pose
            detach_pos, detach_rot = robot_env.detach_sigma()
            j = 0 # Episode timestep

            while True:
                # ===========================================================
                #                   Policy inference loop
                # ===========================================================
                print("=========== Policy inference ============")
                while not robot_env.keyboard.finish and not robot_env.keyboard.discard and not robot_env.keyboard.help:
                    if j >= max_episode_length:
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

                    for step in range(Ta):
                        if step > 0:
                            start_time = time.time()

                        if step > 0:
                            robot_state = robot_env.get_robot_state()
                        
                        # Get absolute action using EpisodeManager
                        deployed_action, gripper_action, curr_p, curr_r, curr_p_action, curr_r_action = episode_manager.get_absolute_action_for_step(
                            action_seq[np.newaxis, :], step
                        )

                        # Get current robot state for recording
                        tcpPose = robot_state['tcp_pose']
                        jointPose = robot_state['joint_pos']
                        
                        # Deploy action using RobotEnv
                        robot_env.deploy_action(deployed_action, gripper_action[0])

                        # Save demonstrations to the buffer
                        wrist_cam.append(robot_state['demo_wrist_img'].permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
                        side_cam.append(robot_state['demo_side_img'].permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
                        tcp_pose.append(tcpPose)
                        joint_pos.append(jointPose)
                        action.append(np.concatenate((curr_p_action, curr_r_action, [gripper_action[0]])))
                        action_mode.append(ROBOT)

                        if step >= Ta - To + 1:
                            img_0 = robot_state['policy_side_img'] / 255.
                            img_1 = robot_state['policy_wrist_img'] / 255.
                            if 'ee_pose' in cfg.shape_meta['obs']:
                                state = robot_state['tcp_pose']
                            else:
                                state = robot_state['joint_pos']
                            episode_manager.update_observation(img_0, img_1, state)

                        time.sleep(max(1 / fps - (time.time() - start_time), 0))
                        j += 1

                # Reset the signal of request for help 
                robot_env.keyboard.help = False
                # Compensate for the transformation of robot tcp pose during sigma detachment
                resume_pos = episode_manager.last_p[0]
                resume_rot = episode_manager.last_r[0]
                translate = resume_pos - detach_pos
                rotation = detach_rot.inv() * resume_rot
                robot_env.sigma.resume()
                robot_env.sigma.transform_from_robot(translate, rotation)

                print("============ Human intervention =============")
                # ============================================================
                #                   Human intervention loop
                # ============================================================
                # Get current pose for human teleoperation
                current_p = episode_manager.last_p[0]
                current_r = episode_manager.last_r[0]
                
                while not robot_env.keyboard.finish and not robot_env.keyboard.discard and not robot_env.keyboard.infer:
                    if j >= max_episode_length:
                        break
                    
                    # Use RobotEnv's human teleoperation step
                    processed_data, current_p, current_r = robot_env.human_teleop_step(current_p, current_r)
                    
                    # If teleoperation is paused (throttle activated), continue
                    if processed_data is None:
                        continue
                    
                    # Save demonstrations to the buffer
                    wrist_cam.append(processed_data['demo_wrist_img'].permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
                    side_cam.append(processed_data['demo_side_img'].permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
                    tcp_pose.append(processed_data['tcp_pose'])
                    joint_pos.append(processed_data['joint_pos'])
                    action.append(processed_data['action'])
                    action_mode.append(processed_data['action_mode'])
                    
                    # Update policy observation buffer using EpisodeManager
                    if 'ee_pose' in cfg.shape_meta['obs']:
                        state = processed_data['tcp_pose']
                    else:
                        state = processed_data['joint_pos']

                    img_0 = processed_data['policy_side_img'] / 255.
                    img_1 = processed_data['policy_wrist_img'] / 255.
                    episode_manager.update_observation(img_0, img_1, state)
                    
                    j += 1

                # Update EpisodeManager pose tracking after human intervention
                episode_manager.last_p = current_p[np.newaxis, :]
                episode_manager.last_r = R.from_quat(current_r.as_quat(scalar_first=True)[np.newaxis, :], scalar_first=True)

                # Reset the signal of request for inference
                robot_env.keyboard.infer = False
                # Detach the teleop device except when human intervention
                detach_pos, detach_rot = robot_env.detach_sigma()

                # This episode fails to accomplish the task
                if j >= max_episode_length or robot_env.keyboard.discard:
                    break

                # Save the demonstrations to the replay buffer
                if robot_env.keyboard.finish:
                    episode = dict()
                    episode['wrist_cam'] = np.stack(wrist_cam, axis=0)
                    episode['side_cam'] = np.stack(side_cam, axis=0)
                    episode['tcp_pose'] = np.stack(tcp_pose, axis=0)
                    episode['joint_pos'] = np.stack(joint_pos, axis=0)
                    episode['action'] = np.stack(action, axis=0)
                    episode['action_mode'] = postprocess_action_mode(np.array(action_mode))
                    replay_buffer.add_episode(episode, compressors='disk')
                    episode_id = replay_buffer.n_episodes - 1
                    print('Saved episode ', episode_id)
                    break

            # Reset robot using RobotEnv
            robot_env.reset_robot()
            print("Reset!")
            
            episode_idx += 1
            
            # Sirius rollout halts if human intervention samples exceed 1 / 3 of original human demonstration
            if np.sum(replay_buffer.data['action_mode'] == INTV) * 3 / num_round >= np.sum(replay_buffer.data['action_mode'] == HUMAN):
                break

            print(f"Current progress: {np.sum(replay_buffer.data['action_mode'] == INTV) * 300 / num_round / np.sum(replay_buffer.data['action_mode'] == HUMAN)} %")

            # For task configuration reset
            time.sleep(5)
    
    finally:
        # Save the replay buffer to a new path
        save_zarr_path = os.path.join(eval_cfg.save_buffer_path, 'replay_buffer.zarr')
        replay_buffer.save_to_path(save_zarr_path)
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
    elif len(device_ids) > 1:
        torch.multiprocessing.spawn(main, args=(cfg, device_ids), nprocs=len(device_ids), join=True)