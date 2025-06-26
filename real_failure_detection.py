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
import tqdm
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R
from collections import defaultdict, OrderedDict

from diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.diffusion_policy.model.common.rotation_transformer import RotationTransformer

from robot_env import RobotEnv, HUMAN, ROBOT, PRE_INTV, INTV, postprocess_action_mode
from failure_detection import FailureDetector
from util.episode_utils import EpisodeManager
from util.image_utils import create_failure_visualization


# Failure detection macros
ACTION_INCONSISTENCY = 1
OT = 2

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
    inconsistency_metric = eval_cfg.inconsistency_metric
    assert inconsistency_metric in ['stat', 'expert']
    num_samples = eval_cfg.num_samples
    num_expert_candidates = eval_cfg.num_expert_candidates
    post_process_action_mode = eval_cfg.post_process_action_mode
    action_inconsistency_percentile = eval_cfg.action_inconsistency_percentile
    ot_percentile = eval_cfg.ot_percentile
    soft_ot_ratio = eval_cfg.soft_ot_ratio

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
    camera_serial = ["135122075425", "135122070361"]
    robot_env = RobotEnv(camera_serial, img_shape, fps)
    
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
        num_samples=num_samples
    )
    
    # Initialize failure detector
    failure_detector = FailureDetector(
        Ta=Ta,
        action_inconsistency_percentile=action_inconsistency_percentile,
        ot_percentile=ot_percentile
    )
    failure_detector.start_async_processing()
    
    # Initialize demonstration buffer
    zarr_path = os.path.join(eval_cfg.train_dataset_path, 'replay_buffer.zarr')
    replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=None)
    if 'action_mode' not in replay_buffer.keys():
        replay_buffer.data['action_mode'] = np.full((replay_buffer.n_steps, ), HUMAN)
    if 'failure_indices' not in replay_buffer.keys():
        replay_buffer.data['failure_indices'] = np.zeros((replay_buffer.n_steps, ), dtype=np.bool_)

    # Distinguish original human demonstrations from rollouts with human intervention
    human_demo_indices = []
    for i in range(replay_buffer.n_episodes):
        episode_start = replay_buffer.episode_ends[i-1] if i > 0 else 0
        if np.any(replay_buffer.data['action_mode'][episode_start: replay_buffer.episode_ends[i]] == HUMAN):
            human_demo_indices.append(i)
            
    all_human_latent = []
    human_eps_len = []

    for i in tqdm.tqdm(human_demo_indices, desc="Obtaining latent for human demo"):
        human_episode = replay_buffer.get_episode(i)
        eps_side_img = (torch.from_numpy(human_episode['side_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_wrist_img = (torch.from_numpy(human_episode['wrist_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_state = np.zeros((human_episode['tcp_pose'].shape[0], ee_pose_dim))
        eps_state[:, :3] = human_episode['tcp_pose'][:, :3]
        eps_state[:, 3:] = obs_rot_transformer.forward(human_episode['tcp_pose'][:, 3:])
        eps_state = (torch.from_numpy(eps_state)).to(device)
        demo_len = human_episode['action'].shape[0]

        human_latent = torch.zeros((demo_len // Ta, int(To*obs_feature_dim)), device=device)
        for idx in range(demo_len // Ta):
            human_demo_idx = idx * Ta
            if human_demo_idx < To - 1:
                indices = [0] * (To-1-human_demo_idx) + list(range(human_demo_idx+1))
                obs_dict = {
                    'side_img': eps_side_img[indices, :].unsqueeze(0),
                    'wrist_img': eps_wrist_img[indices, :].unsqueeze(0), 
                    'ee_pose': eps_state[indices, :].unsqueeze(0)
                }
            else:
                obs_dict = {
                    'side_img': eps_side_img[human_demo_idx-To+1: human_demo_idx+1, :].unsqueeze(0),
                    'wrist_img': eps_wrist_img[human_demo_idx-To+1: human_demo_idx+1, :].unsqueeze(0), 
                    'ee_pose': eps_state[human_demo_idx-To+1: human_demo_idx+1, :].unsqueeze(0)
                }

            with torch.no_grad():
                obs_features = policy.extract_latent(obs_dict)
                human_latent[idx] = obs_features.squeeze(0).reshape(-1)

        human_eps_len.append(eps_side_img.shape[0])
        all_human_latent.append(human_latent)
        
    # Initialize failure-detection-related hyperparameters
    match_round = re.search(r'round(\d)', eval_cfg.train_dataset_path)
    training_set_num_round = int(match_round.group(1)) if match_round else 0

    if training_set_num_round != num_round: # Re-initialize buffers for the current policy
        print("Re-initializing success statistics for the current round.")
    else: # Load the previous success statistics
        print("Loading success statistics from the previous round.")
        prev_success_states = np.load(os.path.join(eval_cfg.train_dataset_path, 'success_stats.npz'))
        failure_detector.load_success_statistics(prev_success_states)
        
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
            action_inconsistency_buffer = []

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
                    robot_state['side_img'] / 255.0,
                    robot_state['wrist_img'] / 255.0,
                    robot_state['tcp_pose'] if state_type == 'ee_pose' else robot_state['joint_pos']
                )
            
            # Initialize the action pose tracking
            if eval_cfg.random_init and random_init_pose is not None:
                episode_manager.initialize_pose(random_init_pose[:3], random_init_pose[3:])
            else:
                episode_manager.initialize_pose(robot_env.robot.init_pose[:3], robot_env.robot.init_pose[3:])
                
            # Match the current rollout with the closest expert episode by initial visual latent vector
            rollout_init_latent = episode_manager.extract_latent().unsqueeze(0)
            candidate_expert_indices = episode_manager.find_matching_expert_demo(
                rollout_init_latent, 
                all_human_latent, 
                human_demo_indices,
                num_expert_candidates
            )
            
            matched_human_idx = human_demo_indices[candidate_expert_indices[0]]
            human_latent = all_human_latent[matched_human_idx]
            demo_len = human_eps_len[matched_human_idx]

            # Initialize the rollout optimal transport components
            expert_weight = torch.ones((demo_len // Ta,), device=device) / float(demo_len // Ta)
            expert_indices = torch.arange(demo_len // Ta, device=device)
            greedy_ot_plan = torch.zeros((demo_len // Ta, max_episode_length // Ta), device=device)
            greedy_ot_cost = torch.zeros((max_episode_length // Ta,), device=device)
            rollout_latent = torch.zeros((max_episode_length // Ta, int(To*obs_feature_dim)), device=device)

            # Scene alignment
            if save_img or not os.path.isfile(os.path.join(output_dir, f"side_{episode_idx}.png")):
                robot_env.save_scene_images(output_dir, episode_idx)
            else:
                robot_env.align_scene_with_file(output_dir, episode_idx)
            
            # Detach the teleop device for initial robot control
            detach_pos, detach_rot = robot_env.detach_sigma()
            
            j = 0  # Episode timestep
            failure_logs = OrderedDict()
            
            while True:
                if j >= max_episode_length - Ta:
                    print("Maximum episode length reached, turning to human for help.")
                    robot_env.keyboard.help = True
                
                # ===========================================================
                #                   Policy inference loop
                # ===========================================================
                print("=========== Policy inference ============")
                while not robot_env.keyboard.finish and not robot_env.keyboard.discard and not robot_env.keyboard.help:
                    # Get robot state and observations
                    robot_state = robot_env.get_robot_state()
                    
                    # Update observation history
                    episode_manager.update_observation(
                        robot_state['side_img'] / 255.0,
                        robot_state['wrist_img'] / 255.0,
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
                        start_time = time.time()  # Track time for fps control

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
                        wrist_cam.append(state_data['wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                        side_cam.append(state_data['side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                        tcp_pose.append(state_data['tcp_pose'])
                        joint_pos.append(state_data['joint_pos'])
                        action.append(np.concatenate((curr_p_action, curr_r_action, [gripper_action[0]])))
                        action_mode.append(ROBOT)
                        
                        # Update policy observation if needed
                        if step >= Ta - To + 1:
                            episode_manager.update_observation(
                                state_data['side_img'] / 255.0,
                                state_data['wrist_img'] / 255.0,
                                state_data['tcp_pose'] if state_type == 'ee_pose' else state_data['joint_pos']
                            )
                        
                        time.sleep(max(1 / fps - (time.time() - start_time), 0))
                        j += 1

                    action_inconsistency = episode_manager.imagine_future_action(action_seq, predicted_abs_actions)
                    
                    # Track action inconsistency for failure detection
                    action_inconsistency_buffer.extend([action_inconsistency] * Ta)
                    
                    # Update rollout latent for this timestep
                    idx = j // Ta - 1
                    rollout_latent[idx] = curr_latent
                    
                    # Submit OT matching task asynchronously
                    candidate_expert_latents = [all_human_latent[i] for i in candidate_expert_indices]
                    failure_detector.submit_ot_matching_task(
                        rollout_latent=rollout_latent.clone(),
                        idx=idx, 
                        candidate_expert_latents=candidate_expert_latents,
                        candidate_expert_indices=candidate_expert_indices,
                        human_demo_indices=human_demo_indices,
                        all_human_latent=all_human_latent,
                        human_eps_len=human_eps_len,
                        replay_buffer=replay_buffer,
                        device=device,
                        max_episode_length=max_episode_length
                    )
                    
                    # Process any available async results
                    failure_flag = False
                    failure_reason = None
                    failure_type = None
                    
                    # Get results from the failure detector
                    results = failure_detector.get_results()
                    for result in results:
                        if result["task_type"] == "ot_matching" and result["idx"] <= idx:
                            # Update with the latest OT results
                            matched_human_idx = result["matched_human_idx"]
                            human_latent = result["human_latent"]
                            demo_len = result["demo_len"]
                            eps_side_img = result["eps_side_img"]
                            expert_weight = result["expert_weight"]
                            expert_indices = result["expert_indices"]
                            greedy_ot_plan = result["greedy_ot_plan"]
                            greedy_ot_cost = result["greedy_ot_cost"]
                            
                            # Submit failure detection task
                            failure_detector.submit_failure_detection_task(
                                action_inconsistency_buffer=action_inconsistency_buffer[:int(result["idx"]+1)*Ta].copy(),
                                idx=result["idx"],
                                greedy_ot_cost=greedy_ot_cost.clone(),
                                greedy_ot_plan=greedy_ot_plan.clone(),
                                max_episode_length=max_episode_length
                            )
                        
                        elif result["task_type"] == "failure_detection" and result["idx"] <= idx:
                            # Update with failure detection results
                            failure_flag = result["failure_flag"]
                            failure_reason = result["failure_reason"]
                            if failure_reason == "action inconsistency":
                                failure_type = ACTION_INCONSISTENCY
                            elif failure_reason == "OT violation":
                                failure_type = OT
                    
                    if failure_flag or j >= max_episode_length - Ta:
                        if failure_flag:
                            print(f"Failure detected! Due to {failure_reason}")
                        else:
                            print("Maximum episode length reached!")
                            
                        print("Press 'c' to continue; Press 'd' to discard the demo; Press 'h' to request human intervention; Press 'f' to finish the episode.")
                        while not robot_env.keyboard.ctn and not robot_env.keyboard.discard and not robot_env.keyboard.help and not robot_env.keyboard.finish:
                            time.sleep(0.1)
                            
                        if failure_flag and not robot_env.keyboard.finish:
                            failure_logs[idx] =  failure_type
                            
                        if robot_env.keyboard.ctn and j < max_episode_length - Ta:
                            print("False Positive failure! Continue policy rollout.")
                            robot_env.keyboard.ctn = False
                            # Temporarily ignore threshold if action inconsistency triggered false positive
                            if failure_reason == "action inconsistency":
                                failure_detector.expert_action_threshold = np.inf
                            continue
                        elif robot_env.keyboard.ctn and j >= max_episode_length - Ta:
                            print("Cannot continue policy rollout, maximum episode length reached. Calling for human intervention.")
                            robot_env.keyboard.ctn = False
                            robot_env.keyboard.help = True
                            break

                # Ensure we have final OT results before human intervention
                failure_detector.submit_ot_matching_task(
                    rollout_latent=rollout_latent.clone(),
                    idx=idx,
                    candidate_expert_latents=candidate_expert_latents,
                    candidate_expert_indices=candidate_expert_indices,
                    human_demo_indices=human_demo_indices,
                    all_human_latent=all_human_latent,
                    human_eps_len=human_eps_len,
                    replay_buffer=replay_buffer,
                    device=device,
                    max_episode_length=max_episode_length
                )
                
                # Wait for final results and empty queues
                failure_detector.empty_queue()
                results = failure_detector.wait_for_final_results()
                for result in results:
                    if result["task_type"] == "ot_matching":
                        # Update with the latest OT results
                        matched_human_idx = result["matched_human_idx"]
                        human_latent = result["human_latent"]
                        demo_len = result["demo_len"]
                        eps_side_img = result["eps_side_img"]
                        expert_weight = result["expert_weight"]
                        expert_indices = result["expert_indices"]
                        greedy_ot_plan = result["greedy_ot_plan"]
                        greedy_ot_cost = result["greedy_ot_cost"]
                failure_detector.empty_result_queue()
                
                # Process human intervention if requested
                if robot_env.keyboard.help:
                    # Reset the signal of request for help 
                    robot_env.keyboard.help = False
                    
                    # Rewind the robot while human resets the environment
                    curr_timestep = j
                    soft_ot_threshold = soft_ot_ratio * torch.sum(greedy_ot_cost[:curr_timestep // Ta])
                    curr_tcp = predicted_abs_actions[0, Ta-1, :7] # Use the last target action for rewinding instead of the current robot state
                    curr_pos = curr_tcp[:3]
                    curr_rot = R.from_quat(curr_tcp[3:], scalar_first=True)
                    for i in range(curr_timestep):
                        # Adaptively check if OT cost dropped below soft threshold to stop rewinding
                        if j % Ta == 0 and j > 0:
                            if torch.sum(greedy_ot_cost[:j // Ta]) < soft_ot_threshold:
                                print("OT cost dropped below the soft threshold, stop rewinding.")
                                break
                            
                            # Rewind the OT plan
                            recovered_expert_weight = torch.zeros((demo_len // Ta,), device=device)
                            recovered_expert_weight[expert_indices] = expert_weight.to(recovered_expert_weight.dtype)
                            expert_weight = recovered_expert_weight + greedy_ot_plan[:, j // Ta - 1]
                            expert_indices = torch.nonzero(expert_weight)[:, 0]
                            greedy_ot_plan[:, j // Ta - 1] = 0.
                            greedy_ot_cost[j // Ta - 1] = 0.
                            
                            # Adjust failure indices if needed
                            if len(failure_logs) > 0 and failure_flag:
                                latest_failure_timestep, latest_failure_type = failure_logs.popitem()
                                for timestep, failure_type in failure_logs.items():
                                    if timestep >= latest_failure_timestep - 1:
                                        failure_logs.move_to_end(timestep)
                                        _, _ = failure_logs.popitem()
                                failure_logs[latest_failure_timestep - 1] = latest_failure_type
                        
                        # Get previous action data to rewind
                        prev_wrist_cam = wrist_cam.pop()
                        prev_side_cam = side_cam.pop()
                        tcp_pose.pop()
                        joint_pos.pop()
                        prev_action = action.pop()
                        action_mode.pop()
                        action_inconsistency_buffer.pop()
                        
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
                        teleop_data['side_img'] / 255.0,
                        teleop_data['wrist_img'] / 255.0,
                        teleop_data['tcp_pose'] if state_type == 'ee_pose' else teleop_data['joint_pos']
                    )
                    
                    # Store demo data
                    wrist_cam.append(teleop_data['wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    side_cam.append(teleop_data['side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    tcp_pose.append(teleop_data['tcp_pose'])
                    joint_pos.append(teleop_data['joint_pos'])
                    action.append(teleop_data['action'])
                    action_mode.append(teleop_data['action_mode'])
                    action_inconsistency_buffer.append(0)  # No inconsistency for human actions
                    
                    j += 1
                
                # Reset pose for policy after human intervention
                episode_manager.initialize_pose(last_p, last_r.as_quat(scalar_first=True))
                
                # Reset signals
                robot_env.keyboard.infer = False
                detach_pos, detach_rot = robot_env.detach_sigma()

                # If episode discarded, break to next episode
                if robot_env.keyboard.discard:
                    break

                # Identify if the rollout is successful
                success = robot_env.keyboard.finish and INTV not in action_mode
                
                # Update thresholds based on episode outcome
                if success:
                    success_action_inconsistency = np.array(action_inconsistency_buffer).sum()
                    failure_detector.update_thresholds(
                        success_action_inconsistency=success_action_inconsistency,
                        greedy_ot_cost=greedy_ot_cost, 
                        timesteps=j//Ta
                    )
                    if len(failure_logs) > 0: # False positive
                        ot_fp = OT in failure_logs.values()
                        action_fp = ACTION_INCONSISTENCY in failure_logs.values()
                        assert ot_fp or action_fp
                        failure_detector.update_percentile_fp(ot_fp=ot_fp, action_fp=action_fp)
                        
                        print("False positive trajectory! Lowering percentiles...")
                    
                    print(f"Updated thresholds: action={failure_detector.expert_action_threshold}, " 
                    f"OT={failure_detector.expert_ot_threshold}")
                else:
                    if len(failure_logs) == 0: # False negative
                        failure_detector.update_percentile_fn()
                        print("False negative trajectory! Raising percentiles...")
                        print(f"Updated thresholds: action={failure_detector.expert_action_threshold}, " 
                              f"OT={failure_detector.expert_ot_threshold}")
                
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
                    if len(failure_logs) > 0:
                        failure_signal[list(failure_logs.keys())] = 1
                    episode['failure_indices'] = np.repeat(failure_signal, Ta)
                    
                    # Add episode to replay buffer
                    replay_buffer.add_episode(episode, compressors='disk')
                    episode_id = replay_buffer.n_episodes - 1
                    print('Saved episode ', episode_id)
                    
                    # Create visualization
                    action_inconsistency_buffer = np.array(action_inconsistency_buffer)
                    fig = create_failure_visualization(
                        action_inconsistency_buffer=action_inconsistency_buffer,
                        greedy_ot_plan=greedy_ot_plan,
                        greedy_ot_cost=greedy_ot_cost,
                        human_eps_len=human_eps_len[matched_human_idx],
                        max_episode_length=max_episode_length,
                        Ta=Ta,
                        episode=episode,
                        eps_side_img=eps_side_img,
                        demo_len=demo_len,
                        failure_indices=list(failure_logs.keys())
                    )
                    plt.savefig(f'{eval_cfg.save_buffer_path}/episode_{episode_id}.png', bbox_inches='tight')
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
        # Stop async processing
        failure_detector.stop_async_processing()
        
        # Save the replay buffer
        save_zarr_path = os.path.join(eval_cfg.save_buffer_path, 'replay_buffer.zarr')
        replay_buffer.save_to_path(save_zarr_path)
        
        # Save success statistics
        success_stats = failure_detector.get_success_statistics()
        np.savez(os.path.join(eval_cfg.save_buffer_path, 'success_stats.npz'), **success_stats)
        
        print("Saved replay buffer and success statistics to", save_zarr_path)
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