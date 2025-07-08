import pathlib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diffusion_policy"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "FAIL_DETECT"))
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import re
import hydra
import torch
import numpy as np
import dill
import time
import tqdm
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R

from diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer

from robot_env import RobotEnv, HUMAN, ROBOT, PRE_INTV, INTV, postprocess_action_mode
from hardware.my_device.macros import CAM_SERIAL
from util.episode_utils import EpisodeManager
from FAIL_DETECT.eval import get_baseline_model, logpZO_UQ


def main(rank, eval_cfg, device_ids):
    fps = 10
    world_size = len(device_ids)
    device_id = device_ids[rank]
    device = f"cuda:{device_id}"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

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

    # get baseline model
    baseline_model = get_baseline_model(eval_cfg.baseline_model_path, device=device)
    baseline_normalizer = torch.load(eval_cfg.baseline_normalizer_path)
    baseline_normalizer.to(device)
    logpZO_upper_bound = np.load(eval_cfg.baseline_stats_path)['target_traj']

    # Extract some hyperparameters from the config
    To = policy.n_obs_steps
    Ta = policy.n_action_steps
    obs_feature_dim = policy.obs_feature_dim
    img_shape = cfg.task['shape_meta']['obs']['wrist_img']['shape']
    if 'ee_pose' in cfg.shape_meta['obs']:
        state_shape = cfg.task['shape_meta']['obs']['ee_pose']['shape']
    else:
        state_shape = cfg.task['shape_meta']['obs']['qpos']['shape']

    # Random seed for every rollout
    seed = eval_cfg.seed
    np.random.seed(seed)

    # Failure detection hyperparameters
    assert eval_cfg.Ta <= Ta
    Ta = eval_cfg.Ta
    post_process_action_mode = eval_cfg.post_process_action_mode

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
    
    # Initialize policy interaction helper
    episode_manager = EpisodeManager(
        policy=policy,
        obs_rot_transformer=obs_rot_transformer,
        action_rot_transformer=action_rot_transformer,
        obs_feature_dim=obs_feature_dim,
        img_shape=img_shape,
        state_type=state_type,
        state_shape=state_shape,
        action_dim=action_dim,
        To=To,
        Ta=Ta,
        device=device,
        num_samples=1
    )
    
    # Initialize demonstration buffer
    zarr_path = os.path.join(eval_cfg.train_dataset_path, 'replay_buffer.zarr')
    replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=None)
    if 'action_mode' not in replay_buffer.keys():
        replay_buffer.data['action_mode'] = np.full((replay_buffer.n_steps, ), HUMAN)
    if 'failure_indices' not in replay_buffer.keys():
        replay_buffer.data['failure_indices'] = np.zeros((replay_buffer.n_steps, ), dtype=np.bool_)
        
    # Initialize failure-detection-related hyperparameters
    match_round = re.search(r'round(\d)', eval_cfg.train_dataset_path)

    # Distinguish original human demonstrations from rollouts with human intervention
    human_demo_indices = []
    for i in range(replay_buffer.n_episodes):
        episode_start = replay_buffer.episode_ends[i-1] if i > 0 else 0
        if np.any(replay_buffer.data['action_mode'][episode_start: replay_buffer.episode_ends[i]] == HUMAN):
            human_demo_indices.append(i)
    human_eps_len = []

    for i in human_demo_indices:
        human_episode = replay_buffer.get_episode(i)
        eps_side_img = (torch.from_numpy(human_episode['side_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        human_eps_len.append(eps_side_img.shape[0])
        
    # Evaluation starts here
    episode_idx = 0
    max_episode_length = int(torch.max(torch.tensor(human_eps_len)) // Ta * Ta)  # Define the maximum episode length
    os.makedirs(eval_cfg.save_buffer_path, exist_ok=True)
    
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

            # Reset the robot to home pose
            random_init_pose = None
            if eval_cfg.random_init:
                random_init_pose = robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
            
            robot_state = robot_env.reset_robot(eval_cfg.random_init, random_init_pose)
            
            # Initialize policy observation history
            episode_manager.reset_observation_history()
            
            # Update initial observations
            for _ in range(To):
                episode_manager.update_observation(
                    robot_state['policy_side_img'] / 255.0,
                    robot_state['policy_wrist_img'] / 255.0,
                    robot_state['tcp_pose'] if state_type == 'ee_pose' else robot_state['joint_pos']
                )
            
            # Initialize the action pose tracking
            if eval_cfg.random_init and random_init_pose is not None:
                episode_manager.initialize_pose(random_init_pose[:3], random_init_pose[3:])
            else:
                episode_manager.initialize_pose(robot_env.robot.init_pose[:3], robot_env.robot.init_pose[3:])

            # Scene alignment
            if save_img or not os.path.isfile(os.path.join(output_dir, f"side_{episode_idx}.png")):
                robot_env.save_scene_images(output_dir, episode_idx)
            else:
                robot_env.align_scene_with_file(output_dir, episode_idx)
            
            # Detach the teleop device for initial robot control
            detach_pos, detach_rot = robot_env.detach_sigma()
            
            j = 0  # Episode timestep
            failure_indices = []
            
            while True:
                if j >= max_episode_length - Ta:
                    print("Maximum episode length reached, turning to human for help.")
                    robot_env.keyboard.help = True
                
                # ===========================================================
                #                   Policy inference loop
                # ===========================================================
                print("=========== Policy inference ============")
                while not robot_env.keyboard.finish and not robot_env.keyboard.discard and not robot_env.keyboard.help:
                    start_time = time.time()  # Track time for fps control
                    # Get robot state and observations
                    robot_state = robot_env.get_robot_state()
                    
                    # Update observation history
                    episode_manager.update_observation(
                        robot_state['policy_side_img'] / 255.0,
                        robot_state['policy_wrist_img'] / 255.0,
                        robot_state['tcp_pose'] if state_type == 'ee_pose' else robot_state['joint_pos']
                    )
                    
                    # Get action sequence for execution
                    policy_obs = episode_manager.get_policy_observation()
                    with torch.no_grad():
                        curr_action, curr_latent = policy.predict_action(policy_obs, return_latent=True)
                        curr_latent = curr_latent[0].reshape(-1)
                        
                    # Get first Ta actions and execute on robot
                    np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
                    action_seq = np_action_dict['action']

                    # Convert to absolute actions
                    predicted_abs_actions = np.zeros_like(action_seq[:, :, :8])
                    
                    for step in range(Ta):
                        if step > 0:
                            start_time = time.time()
                            
                        # Get robot state
                        if step == 0:
                            state_data = robot_state
                        else:
                            state_data = robot_env.get_robot_state()
                        
                        # Get absolute action for this step
                        deployed_action, gripper_action, curr_p, curr_r, curr_p_action, curr_r_action = \
                            episode_manager.get_absolute_action_for_step(action_seq, step)
                        predicted_abs_actions[:, step] = np.concatenate((curr_p, curr_r.as_quat(scalar_first=True), gripper_action[:, np.newaxis]), -1)
                        
                        # Execute action on robot
                        robot_env.deploy_action(deployed_action, gripper_action[0])
                        
                        # Save to episode buffers
                        wrist_cam.append(state_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                        side_cam.append(state_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                        tcp_pose.append(state_data['tcp_pose'])
                        joint_pos.append(state_data['joint_pos'])
                        action.append(np.concatenate((curr_p_action, curr_r_action, [gripper_action[0]])))
                        action_mode.append(ROBOT)
                        
                        # Update policy observation if needed
                        if step >= Ta - To + 1:
                            episode_manager.update_observation(
                                state_data['policy_side_img'] / 255.0,
                                state_data['policy_wrist_img'] / 255.0,
                                state_data['tcp_pose'] if state_type == 'ee_pose' else state_data['joint_pos']
                            )
                        
                        time.sleep(max(1 / fps - (time.time() - start_time), 0))
                        j += 1

                    # FAIL-DETECT: Baseline failure detection module
                    normalized_obs = baseline_normalizer.normalize(policy_obs)
                    this_nobs = dict_apply(normalized_obs, lambda x: x[:, :policy.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
                    nobs_features = policy.obs_encoder.get_dense_feats(this_nobs)
                    global_cond = nobs_features.reshape(1, -1)
                    curr_logp = logpZO_UQ(baseline_model, global_cond, None)
                    print(curr_logp.item(), logpZO_upper_bound[j // Ta - 1])
                    failure_flag = curr_logp.item() > logpZO_upper_bound[j // Ta - 1] if len(logpZO_upper_bound) >= j // Ta else False
                    if failure_flag or j >= max_episode_length - Ta:
                        if failure_flag:
                            print("Failure detected!")
                        else: 
                            print("Maximum episode length reached!")

                        print("Press 'c' to continue; Press 'd' to discard the demo; Press 'h' to request human intervention; Press 'f' to finish the episode.")
                        while not robot_env.keyboard.ctn and not robot_env.keyboard.discard and not robot_env.keyboard.help and not robot_env.keyboard.finish:
                            time.sleep(0.1)
                            
                        if failure_flag and not robot_env.keyboard.finish:
                            failure_indices.append(j // Ta - 1)
                            
                        if robot_env.keyboard.ctn and j < max_episode_length - Ta:
                            print("False Positive failure! Continue policy rollout.")
                            robot_env.keyboard.ctn = False
                            continue
                        elif robot_env.keyboard.ctn and j >= max_episode_length - Ta:
                            print("Cannot continue policy rollout, maximum episode length reached. Calling for human intervention.")
                            robot_env.keyboard.ctn = False
                            robot_env.keyboard.help = True
                            break
                
                # Process human intervention if requested
                if robot_env.keyboard.help:
                    # Reset the signal of request for help 
                    robot_env.keyboard.help = False
                    
                    # Rewind the robot while human resets the environment
                    curr_timestep = j
                    # curr_tcp = robot_env.get_robot_state()['tcp_pose']
                    curr_tcp = predicted_abs_actions[0, Ta-1, :7] # Use the last target action for rewinding instead of the current robot state
                    curr_pos = curr_tcp[:3]
                    curr_rot = R.from_quat(curr_tcp[3:], scalar_first=True)
                    for i in range(curr_timestep):
                        if j % Ta == 0 and j > 0:
                            if i // Ta >= 3:
                                print("Stop rewinding.")
                                break
                        
                        # Get previous action data to rewind
                        prev_wrist_cam = wrist_cam.pop()
                        prev_side_cam = side_cam.pop()
                        tcp_pose.pop()
                        joint_pos.pop()
                        prev_action = action.pop()
                        action_mode.pop()
                        
                        # Rewind robot by applying inverse action
                        curr_pos, curr_rot = robot_env.rewind_robot(
                            curr_pos,
                            curr_rot,
                            prev_action
                        )
                        j -= 1
                    
                    # Prepare for human intervention by showing reference scene
                    if "prev_side_cam" in locals():
                        print("Please reset the scene and press 'c' to go on to human intervention")
                        ref_side_img = cv2.cvtColor(prev_side_cam, cv2.COLOR_RGB2BGR)
                        ref_wrist_img = cv2.cvtColor(prev_wrist_cam, cv2.COLOR_RGB2BGR)
                        robot_env.align_with_reference(ref_side_img, ref_wrist_img)

                        # Remove the reference images and action buffers
                        del prev_wrist_cam, prev_side_cam, prev_action
                        
                    # Get current pose for human teleop
                    last_p = curr_pos
                    last_r = curr_rot

                    # Transform sigma device from current robot pose
                    translate = curr_pos - detach_pos
                    rotation = detach_rot.inv() * curr_rot
                    robot_env.sigma.resume()
                    robot_env.sigma.transform_from_robot(translate, rotation)
                else:
                    last_p = episode_manager.last_p[0]
                    last_r = episode_manager.last_r[0]
                
                print("============ Human intervention =============")
                # ============================================================
                #                   Human intervention loop
                # ============================================================
                while (not robot_env.keyboard.finish and not robot_env.keyboard.discard and not robot_env.keyboard.infer) or j % Ta:
                    # Execute one step of human teleop
                    teleop_data, last_p, last_r = robot_env.human_teleop_step(last_p, last_r)
                    
                    if teleop_data is None:
                        continue
                    
                    # Update observation history with latest state
                    episode_manager.update_observation(
                        teleop_data['policy_side_img'] / 255.0,
                        teleop_data['policy_wrist_img'] / 255.0,
                        teleop_data['tcp_pose'] if state_type == 'ee_pose' else teleop_data['joint_pos']
                    )
                    
                    # Store demo data
                    wrist_cam.append(teleop_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    side_cam.append(teleop_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    tcp_pose.append(teleop_data['tcp_pose'])
                    joint_pos.append(teleop_data['joint_pos'])
                    action.append(teleop_data['action'])
                    action_mode.append(teleop_data['action_mode'])
                    
                    j += 1
                
                # Reset pose for policy after human intervention
                episode_manager.initialize_pose(last_p, last_r.as_quat(scalar_first=True))
                
                # Reset signals
                robot_env.keyboard.infer = False
                detach_pos, detach_rot = robot_env.detach_sigma()
                
                # If episode discarded, break to next episode
                if robot_env.keyboard.discard:
                    break
                
                # Save episode data if finished
                if robot_env.keyboard.finish:
                    episode = dict()
                    episode['wrist_cam'] = np.stack(wrist_cam, axis=0)
                    episode['side_cam'] = np.stack(side_cam, axis=0)
                    episode['tcp_pose'] = np.stack(tcp_pose, axis=0)
                    episode['joint_pos'] = np.stack(joint_pos, axis=0)
                    episode['action'] = np.stack(action, axis=0)
                    
                    if post_process_action_mode:
                        episode['action_mode'] = postprocess_action_mode(np.array(action_mode))
                    else:
                        episode['action_mode'] = np.array(action_mode)
                    
                    assert episode['action_mode'].shape[0] % Ta == 0, "A Ta-step chunking is required for the entire demo"
                    
                    # Set failure indices in episode data
                    failure_signal = np.zeros((episode['action_mode'].shape[0] // Ta,), dtype=np.bool_)
                    if len(failure_indices) > 0:
                        failure_signal[failure_indices] = 1
                    episode['failure_indices'] = np.repeat(failure_signal, Ta)
                    
                    # Add episode to replay buffer
                    replay_buffer.add_episode(episode, compressors='disk')
                    episode_id = replay_buffer.n_episodes - 1
                    print('Saved episode ', episode_id)
                    break
            
            # Reset robot between episodes
            robot_env.reset_robot()
            print("Reset!")
            
            episode_idx += 1
            
            # Check if we should stop based on intervention ratio
            print(f"Current human intervention ratio: {np.sum(replay_buffer.data['action_mode'] == INTV) * 100 / np.sum(replay_buffer.data['action_mode'] != HUMAN)} %")
            if np.sum(replay_buffer.data['action_mode'] == INTV) * 3 / num_round >= np.sum(replay_buffer.data['action_mode'] == HUMAN):
                break
                
            print(f"Current progress: {np.sum(replay_buffer.data['action_mode'] == INTV) * 300 / num_round / np.sum(replay_buffer.data['action_mode'] == HUMAN)} %")
            
            # For scene configuration reset
            time.sleep(5)
    
    finally:
        
        # Save the replay buffer
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