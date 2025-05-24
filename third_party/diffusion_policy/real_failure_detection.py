import pathlib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
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
import cv2
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms import InterpolationMode
import pygame
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import BlendedGenericTransform

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from hardware.my_device.robot import FlexivRobot, FlexivGripper
from hardware.my_device.camera import CameraD400
from hardware.my_device.keyboard import Keyboard
from hardware.my_device.sigma import Sigma7
from hardware.my_device.logitechG29_wheel import Controller

from util.ot_util import *

camera_serial = ["135122075425", "135122070361"]

# Sirius-specific macros
HUMAN = 0
ROBOT = 1
PRE_INTV = 2
INTV = 3
INTV_STEPS = 15

def process_image(array, size, highlight=False, color = [255, 0, 0]):
    img = Image.fromarray(array).convert('RGBA')
    img = img.resize(size)
    if highlight:
        border_width = 15
        arr = np.array(img)
        arr[:border_width, :] = color + [255]
        arr[-border_width:, :] = color + [255]
        arr[:, :border_width] = color + [255]
        arr[:, -border_width:] = color + [255]
        return Image.fromarray(arr)
    return img

def postprocess_action_mode(action_mode: np.ndarray):
    # Postprocessing for action mode identification
    for i in range(1, len(action_mode)):
        if action_mode[i] == INTV and action_mode[i-1] == ROBOT:
            pre_intv_indices = np.arange(max(0, i-INTV_STEPS), i)
            pre_intv_indices = pre_intv_indices[np.where(action_mode[pre_intv_indices] == INTV, False, True)]
            action_mode[pre_intv_indices] = PRE_INTV
    return action_mode

def tensor_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def main(rank, eval_cfg, device_ids):
    fps = 10  # TODO
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
    obs_feature_dim = policy.obs_feature_dim
    img_shape = cfg.task['shape_meta']['obs']['wrist_img']['shape']
    if 'ee_pose' in cfg.shape_meta['obs']:
        state_shape = cfg.task['shape_meta']['obs']['ee_pose']['shape']
    else:
        state_shape = cfg.task['shape_meta']['obs']['qpos']['shape']

    BICUBIC = InterpolationMode.BICUBIC
    # image_processor = Compose([Resize(img_shape[1:], interpolation=BICUBIC)])
    side_image_processor = Compose([
        Resize((img_shape[1]+8, img_shape[2]+8), interpolation=BICUBIC),
        CenterCrop((img_shape[1], img_shape[2]))
    ])
    wrist_image_processor = Resize((img_shape[1], img_shape[2]), interpolation=BICUBIC)

    # Random seed for every rollout
    seed = int(time.time())
    np.random.seed(seed)

    # Failure detection hyparameters
    assert eval_cfg.Ta <= Ta
    Ta = eval_cfg.Ta
    ot_weight = eval_cfg.ot_weight
    num_samples = eval_cfg.num_samples
    window_size = eval_cfg.window_size
    failure_threshold = eval_cfg.failure_threshold
    
    # Inspect the current round (Sirius-specific)
    match_round = re.search(r'round(\d)', eval_cfg.save_buffer_path)
    assert match_round
    num_round = int(match_round.group(1))
    
    # Initialize hardware
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    cameras = [CameraD400(s) for s in camera_serial]
    keyboard = Keyboard()
    sigma = Sigma7()
    pygame.init()
    controller = Controller(0)

    # Initialize demonstration buffer
    zarr_path = os.path.join(eval_cfg.train_dataset_path, 'replay_buffer.zarr')
    dataset_keys = ['wrist_cam', 'side_cam', 'joint_pos', 'action', 'tcp_pose']
    if 'round' in eval_cfg.train_dataset_path:
        dataset_keys.append('action_mode')
        replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=dataset_keys)
    else:
        replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=dataset_keys)
        replay_buffer.data['action_mode'] = np.full((replay_buffer.n_steps, ), HUMAN)

    # Distinguish original human demonstrations from rollouts with human intervention
    human_demo_indices = []
    for i in range(replay_buffer.n_episodes):
        episode_start = replay_buffer.episode_ends[i-1] if i > 0 else 0
        if np.any(replay_buffer.data['action_mode'][episode_start: replay_buffer.episode_ends[i]] == HUMAN):
            human_demo_indices.append(i)
            
    human_init_latent = torch.zeros((len(human_demo_indices), int(To*obs_feature_dim)), device=device)

    for i in tqdm.tqdm(human_demo_indices, desc="Obtaining latent for human demo"):
        human_episode = replay_buffer.get_episode(i)
        eps_side_img = (torch.from_numpy(human_episode['side_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_wrist_img = (torch.from_numpy(human_episode['wrist_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
        eps_state = np.zeros((human_episode['tcp_pose'].shape[0], ee_pose_dim))
        eps_state[:, :3] = human_episode['tcp_pose'][:, :3]
        eps_state[:, 3:] = obs_rot_transformer.forward(human_episode['tcp_pose'][:, 3:])
        eps_state = (torch.from_numpy(eps_state)).to(device)
        
        indices = [0] * To
        obs_dict = {
            'side_img': eps_side_img[indices, :].unsqueeze(0),
            'wrist_img': eps_wrist_img[indices, :].unsqueeze(0), 
            'ee_pose': eps_state[indices, :].unsqueeze(0)
        }
        
        with torch.no_grad():
            obs_features = policy.extract_latent(obs_dict)
            human_init_latent[i] = obs_features.squeeze(0).reshape(-1)

    # Evaluation starts here
    episode_idx = 0
    max_episode_length = 300
    # episode_list = [x for x in range(eval_cfg.num_episode) if (x + 1) % world_size == rank]

    os.makedirs(eval_cfg.save_buffer_path, exist_ok=True)

    try:
        # for episode_idx in range(eval_cfg.num_episode):
        while True: 
            if keyboard.quit:
                break
            # if episode_idx not in episode_list:
            #     continue
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
            action_mode = []
            wrist_cam = []
            side_cam = []
            action_inconsistency_buffer = []

            # Reset the robot to home pose
            if eval_cfg.random_init:
                random_init_pose = robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
                robot.send_tcp_pose(random_init_pose)
            else:
                robot.send_tcp_pose(robot.init_pose)
            time.sleep(2)
            gripper.move(gripper.max_width)
            time.sleep(0.5)
            print("Reset!")
            # Reset the sigma pose as well
            sigma.reset()
            if eval_cfg.random_init:
                random_p_drift = random_init_pose[:3] - robot.init_pose[:3]
                random_r_drift = R.from_quat(robot.init_pose[3:7], scalar_first=True).inv() * R.from_quat(random_init_pose[3:7], scalar_first=True)
                sigma.transform_from_robot(random_p_drift, random_r_drift)

            # Initialize obs history buffer
            policy_img_0_history = torch.zeros((To, *img_shape), device=device)
            policy_img_1_history = torch.zeros((To, *img_shape), device=device)
            policy_state_history = torch.zeros((To, *state_shape), device=device)

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
            original_img_0 = side_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
            original_img_1 = wrist_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
            policy_img_0 = original_img_0 / 255.
            policy_img_1 = original_img_1 / 255.

            for idx in range(policy_state_history.shape[0]):
                policy_img_0_history[idx] = policy_img_0
                policy_img_1_history[idx] = policy_img_1
                policy_state_history[idx] = state

            # Match the current rollout with the closest expert episode by initial visual latent vector
            curr_obs = {
                'side_img': policy_img_0_history.unsqueeze(0),
                'wrist_img': policy_img_1_history.unsqueeze(0),
                state_type: policy_state_history.unsqueeze(0)
            }

            with torch.no_grad():
                obs_features = policy.extract_latent(curr_obs)
                rollout_latent = obs_features.reshape(-1).unsqueeze(0)
            
            dist_mat = cosine_distance(human_init_latent, rollout_latent).to(device).detach()
            matched_human_idx = human_demo_indices[torch.argmin(dist_mat, dim=0).item()]
            matched_human_episode = replay_buffer.get_episode(matched_human_idx)

            # Fetch the latent representations of the matched human episode
            eps_side_img = (torch.from_numpy(matched_human_episode['side_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
            eps_wrist_img = (torch.from_numpy(matched_human_episode['wrist_cam']).permute(0, 3, 1, 2) / 255.0).to(device)
            eps_state = np.zeros((matched_human_episode['tcp_pose'].shape[0], ee_pose_dim))
            eps_state[:, :3] = matched_human_episode['tcp_pose'][:, :3]
            eps_state[:, 3:] = obs_rot_transformer.forward(matched_human_episode['tcp_pose'][:, 3:])
            eps_state = (torch.from_numpy(eps_state)).to(device)
            demo_len = matched_human_episode['action'].shape[0]
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

            # Initialize the rollout optimal transport components with an assumption of maximum episode length [Need verification]
            expert_weight = torch.ones((demo_len // Ta,), device=device) / float(demo_len // Ta)
            expert_indices = torch.arange(demo_len // Ta, device=device)
            greedy_ot_plan = torch.zeros((demo_len // Ta, max_episode_length // Ta), device=device)
            greedy_ot_cost = torch.zeros((max_episode_length // Ta,), device=device)

            # Keep track of pose from the last frame for relative action space
            if eval_cfg.random_init:
                last_p = random_init_pose[np.newaxis, :3].repeat(num_samples, axis=0)
                last_r = R.from_quat(random_init_pose[np.newaxis, 3:7].repeat(num_samples, axis=0), scalar_first=True)
            else:
                last_p = robot.init_pose[np.newaxis, :3].repeat(num_samples, axis=0)
                last_r = R.from_quat(robot.init_pose[np.newaxis, 3:7].repeat(num_samples, axis=0), scalar_first=True)

            # Initialize the action buffer
            last_predicted_abs_actions = None
            
            # Keep track of throttle usage for human intervention (Default to True because the teleop should follow up from arbitrary pose)
            last_throttle = True
            sigma.detach()
            detach_tcp, _, _, _ = robot.get_robot_state()
            detach_pos = np.array(detach_tcp[:3])
            detach_rot = R.from_quat(np.array(detach_tcp[3:]), scalar_first=True)
            j = 0 # Episode timestep

            while True:
                # ===========================================================
                #                   Policy inference loop
                # ===========================================================
                last_predicted_abs_actions = None
                print("=========== Policy inference ============")
                while not keyboard.finish and not keyboard.discard and not keyboard.help:
                    if j >= max_episode_length - Ta:
                        break

                    start_time = time.time()
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
                    img_0 = side_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) / 255.
                    img_1 = wrist_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) / 255.

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
                        state_type: policy_state_history.unsqueeze(0)
                    }

                    policy_obs = {
                        'side_img': policy_img_0_history.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1),
                        'wrist_img': policy_img_1_history.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1),
                        state_type: policy_state_history.unsqueeze(0).repeat(num_samples, 1, 1)
                    }

                    # Predict eef actions
                    with torch.no_grad():
                        curr_action = policy.predict_action(policy_obs)
                        curr_latent = policy.extract_latent(curr_obs).reshape(-1)
                    np_action_dict = dict_apply(curr_action, lambda x: x.detach().to('cpu').numpy())
                    action_seq = np_action_dict['action']
                    predicted_abs_actions = np.zeros_like(action_seq[:, :, :8])

                    # Derive the action chunk
                    for step in range(Ta):
                        if step > 0:
                            start_time = time.time()
                        # Action calculation
                        curr_p_action = action_seq[:, step, :3]
                        curr_p = last_p + curr_p_action
                        curr_r_action = action_rot_transformer.inverse(action_seq[:, step, 3:action_dim-1])
                        action_rot = R.from_quat(curr_r_action, scalar_first=True)
                        curr_r = last_r * action_rot
                        last_p = curr_p
                        last_r = curr_r

                        # Record demo data
                        cam_data = []
                        for camera in cameras:
                            color_image, depth_image = camera.get_data()
                            cam_data.append((color_image, depth_image))
                        tcpPose, jointPose, _, _ = robot.get_robot_state()
                        # demo_gripper_action = 1 if action_seq[step, 7] >= 0.5 else 0
                        demo_gripper_action = action_seq[:, step, -1]

                        deployed_action = np.concatenate((curr_p[0], curr_r[0].as_quat(scalar_first=True)), 0)
                        predicted_abs_actions[:, step] = np.concatenate((curr_p, curr_r.as_quat(scalar_first=True), demo_gripper_action[:, np.newaxis]), -1)

                        robot.send_tcp_pose(deployed_action)
                        # target_width = gripper.max_width if action_seq[step, 7] < 0.5 else 0 # Threshold could be adjusted at inference time
                        gripper.move(demo_gripper_action[0])

                        # Save demonstrations to the buffer
                        wrist_image = wrist_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1][0].copy(), cv2.COLOR_BGR2RGB)).\
                                        permute(2,0,1)).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                        side_image = side_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0][0].copy(), cv2.COLOR_BGR2RGB)).\
                                                    permute(2,0,1)).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                        wrist_cam.append(wrist_image)
                        side_cam.append(side_image)
                        tcp_pose.append(tcpPose)
                        joint_pos.append(jointPose)
                        action.append(np.concatenate((curr_p_action[0], curr_r_action[0], [demo_gripper_action[0]]), 0))
                        action_mode.append(ROBOT)

                        time.sleep(max(1 / fps - (time.time() - start_time), 0))
                        j += 1

                    tmp_last_p = last_p
                    tmp_last_r = last_r
                    
                    # Predict the entire action chunk for failure detection
                    for step in range(Ta, policy.n_action_steps):
                        tmp_p_action = action_seq[:, step, :3]
                        tmp_curr_p = tmp_last_p + tmp_p_action
                        tmp_r_action = action_rot_transformer.inverse(action_seq[:, step, 3:action_dim-1])
                        action_rot = R.from_quat(tmp_r_action, scalar_first=True)
                        tmp_curr_r = tmp_last_r * action_rot
                        tmp_last_p = tmp_curr_p
                        tmp_last_r = tmp_curr_r
                        demo_gripper_action = action_seq[:, step, -1]
                        deployed_action = np.concatenate((tmp_curr_p, tmp_curr_r.as_quat(scalar_first=True), demo_gripper_action[:, np.newaxis]), -1)
                        predicted_abs_actions[:, step] = deployed_action

                    # Calculate the action inconsistency
                    if last_predicted_abs_actions is None:
                        last_predicted_abs_actions = np.concatenate((np.zeros((Ta, 8)), predicted_abs_actions[0, :-Ta]), 0) # Prevent anomalous value in the beginning
                    action_fluctuation = np.mean(np.sum(np.square(np.linalg.norm(predicted_abs_actions[:, 1:] - predicted_abs_actions[:, :-1], axis=-1)), axis=-1)) # Action fluctuation
                    action_inconsistency = np.mean(np.linalg.norm(predicted_abs_actions[:, :-Ta] - last_predicted_abs_actions[np.newaxis, Ta:], axis=-1)) * \
                        np.exp(-action_fluctuation / 0.0005) # Larger fluctuation entails larger inconsistency
                    last_predicted_abs_actions = predicted_abs_actions[0]
                    action_inconsistency_buffer.extend([action_inconsistency] * Ta)

                    # Calculate the Optimal Transport plan
                    rollout_weight = float(1. / (max_episode_length // Ta))
                    dist2expert = cosine_distance(human_latent[expert_indices], curr_latent.unsqueeze(0)).squeeze(-1)
                    idx = j // Ta
                    while rollout_weight > 0:
                        if dist2expert.shape[0] == 0:
                            break
                        nearest_expert = torch.argmin(dist2expert, dim=0)
                        expert_index = expert_indices[nearest_expert]
                        if expert_weight[nearest_expert] < rollout_weight:
                            rollout_weight -= expert_weight[nearest_expert]
                            greedy_ot_plan[expert_index, idx] += expert_weight[nearest_expert]
                            greedy_ot_cost[idx] += expert_weight[nearest_expert] * dist2expert[nearest_expert]
                            # Release related expert from the buffer
                            expert_weight = tensor_delete(expert_weight, nearest_expert)
                            expert_indices = tensor_delete(expert_indices, nearest_expert)
                            dist2expert = tensor_delete(dist2expert, nearest_expert)
                        else:
                            greedy_ot_plan[expert_index, idx] += rollout_weight
                            greedy_ot_cost[idx] += rollout_weight * dist2expert[nearest_expert]
                            expert_weight[nearest_expert] -= rollout_weight
                            rollout_weight = 0
                            if expert_weight[nearest_expert] == 0:
                                expert_weight = tensor_delete(expert_weight, nearest_expert)
                                expert_indices = tensor_delete(expert_indices, nearest_expert)
                                dist2expert = tensor_delete(dist2expert, nearest_expert)

                    window_index_mean = 0 if idx < window_size else torch.mean(ot_weight * greedy_ot_cost[idx-window_size:idx] \
                                                + torch.tensor(action_inconsistency_buffer[idx-window_size:idx], device=device))
                    window_index_std = torch.inf if idx < window_size else torch.std(ot_weight * greedy_ot_cost[idx-window_size:idx] \
                                                + torch.tensor(action_inconsistency_buffer[idx-window_size:idx], device=device))

                    # Check if the failure index exceeds the threshold
                    if action_inconsistency + ot_weight * greedy_ot_cost[idx] > failure_threshold:
                        print("Failure detected!")
                        while not keyboard.ctn and not keyboard.discard and not keyboard.help:
                            time.sleep(0.1)
                        if keyboard.ctn:
                            print("Not a failure! Continue policy rollout.")
                            keyboard.ctn = False
                            continue

                # Reset the signal of request for help 
                keyboard.help = False
                # Compensate for the transformation of robot tcp pose during sigma detachment
                resume_tcp, _, _, _ = robot.get_robot_state()
                resume_pos = np.array(resume_tcp[:3])
                resume_rot = R.from_quat(np.array(resume_tcp[3:]), scalar_first=True)
                translate = resume_pos - detach_pos
                rotation = detach_rot.inv() * resume_rot
                sigma.transform_from_robot(translate, rotation)

                print("============ Human intervention =============")
                last_p = last_p[0]
                last_r = last_r[0]
                # ============================================================
                #                   Human intervention loop
                # ============================================================
                while not keyboard.finish and not keyboard.discard and not keyboard.infer:
                    if j >= max_episode_length - 1:
                        break
                    
                    start_time = time.time()
                    cam_data = []
                    for camera in cameras:
                        color_image, depth_image = camera.get_data()
                        cam_data.append((color_image, depth_image))
                    tcpPose, jointPose, _, _ = robot.get_robot_state()

                    diff_p, diff_r, width = sigma.get_control()
                    diff_p = robot.init_pose[:3] + diff_p
                    diff_r = R.from_quat(robot.init_pose[3:7], scalar_first=True) * diff_r
                    curr_p_action = diff_p - last_p
                    curr_r_action = last_r.inv() * diff_r
                    last_p = diff_p
                    last_r = diff_r

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
                        last_p, last_r, _ = sigma.get_control()
                        last_p = last_p + robot.init_pose[:3]
                        last_r = R.from_quat(robot.init_pose[3:7], scalar_first=True) * last_r
                        continue

                    # Send command.
                    robot.send_tcp_pose(np.concatenate((diff_p, diff_r.as_quat(scalar_first=True)), 0))
                    gripper.move_from_sigma(width)
                    gripper_action = gripper.max_width * width / 1000
                    # gripper_action = 1 if width < 500 else 0

                    # Renew policy obs buffer for the next inference
                    if 'ee_pose' in cfg.shape_meta['obs']:
                        state = tcpPose
                    else:
                        state = jointPose
                    state = np.array(state)

                    if obs_rot_transformer is not None:
                        tmp_state = np.zeros((ee_pose_dim,))
                        tmp_state[:3] = state[:3]
                        tmp_state[3:] = obs_rot_transformer.forward(state[3:])
                        state = tmp_state
                    state = torch.from_numpy(state)

                    img_0 = side_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0][0].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) / 255.
                    img_1 = wrist_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1][0].copy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) / 255.
                    policy_img_0_history[:-1] = policy_img_0_history[1:]
                    policy_img_0_history[-1] = img_0
                    policy_img_1_history[:-1] = policy_img_1_history[1:]
                    policy_img_1_history[-1] = img_1
                    policy_state_history[:-1] = policy_state_history[1:]
                    policy_state_history[-1] = state
                    
                    # Save demonstrations to the buffer
                    wrist_image = wrist_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[1][0].copy(), cv2.COLOR_BGR2RGB)).\
                                        permute(2,0,1)).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                    side_image = side_image_processor(torch.from_numpy(cv2.cvtColor(cam_data[0][0].copy(), cv2.COLOR_BGR2RGB)).\
                                                permute(2,0,1)).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                    wrist_cam.append(wrist_image)
                    side_cam.append(side_image)
                    tcp_pose.append(tcpPose)
                    joint_pos.append(jointPose)
                    action.append(np.concatenate((curr_p_action, curr_r_action.as_quat(scalar_first=True), [gripper_action])))
                    action_mode.append(INTV)

                    time.sleep(max(1 / fps - (time.time() - start_time), 0))
                    j += 1

                    if j % Ta == 0: # Maintain Optimal Transport calculation during human intervention
                        action_inconsistency_buffer.extend([action_inconsistency] * Ta)
                        rollout_weight = float(1. / (max_episode_length // Ta))
                        curr_obs = {
                            'side_img': policy_img_0_history.unsqueeze(0),
                            'wrist_img': policy_img_1_history.unsqueeze(0),
                            state_type: policy_state_history.unsqueeze(0)
                        }
                        with torch.no_grad():
                            curr_latent = policy.extract_latent(curr_obs).reshape(-1)
                        dist2expert = cosine_distance(human_latent[expert_indices], curr_latent.unsqueeze(0)).squeeze(-1)
                        idx = j // Ta
                        while rollout_weight > 0:
                            if dist2expert.shape[0] == 0:
                                break
                            nearest_expert = torch.argmin(dist2expert, dim=0)
                            expert_index = expert_indices[nearest_expert]
                            if expert_weight[nearest_expert] < rollout_weight:
                                rollout_weight -= expert_weight[nearest_expert]
                                greedy_ot_plan[expert_index, idx] += expert_weight[nearest_expert]
                                greedy_ot_cost[idx] += expert_weight[nearest_expert] * dist2expert[nearest_expert]
                                # Release related expert from the buffer
                                expert_weight = tensor_delete(expert_weight, nearest_expert)
                                expert_indices = tensor_delete(expert_indices, nearest_expert)
                                dist2expert = tensor_delete(dist2expert, nearest_expert)
                            else:
                                greedy_ot_plan[expert_index, idx] += rollout_weight
                                greedy_ot_cost[idx] += rollout_weight * dist2expert[nearest_expert]
                                expert_weight[nearest_expert] -= rollout_weight
                                rollout_weight = 0
                                if expert_weight[nearest_expert] == 0:
                                    expert_weight = tensor_delete(expert_weight, nearest_expert)
                                    expert_indices = tensor_delete(expert_indices, nearest_expert)
                                    dist2expert = tensor_delete(dist2expert, nearest_expert)

                # Reset the last action buffer for policy inference
                last_p = last_p[np.newaxis, :].repeat(num_samples, axis=0)
                last_r = R.from_quat(last_r.as_quat(scalar_first=True)[np.newaxis, :].repeat(num_samples, axis=0), scalar_first=True)

                # Reset the signal of request for inference
                keyboard.infer = False
                # Detach the teleop device except when human intervention
                sigma.detach()
                last_throttle = True
                detach_tcp, _, _, _ = robot.get_robot_state()
                detach_pos = np.array(detach_tcp[:3])
                detach_rot = R.from_quat(np.array(detach_tcp[3:]), scalar_first=True)

                # This episode fails to accomplish the task
                if j >= max_episode_length - 1 or keyboard.discard:
                    break

                # Save the demonstrations to the replay buffer
                if keyboard.finish:
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

                    # Visualize action inconsistency and observation optimal transport cost
                    action_inconsistency_buffer = np.array(action_inconsistency_buffer)
                    ot_cost_final = greedy_ot_cost[:len(action_inconsistency_buffer)//Ta].detach().cpu().numpy()
                    cell_size = 1
                    fig = plt.figure(figsize=(episode['wrist_cam'].shape[0] // Ta * cell_size, 3 * cell_size+2))
                    gs = plt.GridSpec(3, 1, height_ratios=[0.33, 0.33, 0.33], hspace=0.8)
                    # ax = fig.add_subplot(111)
                    action_ax = fig.add_subplot(gs[0])
                    ot_ax = fig.add_subplot(gs[1])
                    final_ax = fig.add_subplot(gs[2])

                    im = action_ax.imshow(action_inconsistency_buffer[::Ta].reshape(1, -1), cmap='plasma', aspect='auto')
                    action_ax.set_xticks([])
                    action_ax.set_yticks([])
                    plt.colorbar(im, ax=action_ax, shrink=0.9)

                    im = ot_ax.imshow(ot_cost_final.reshape(1, -1), cmap='cividis', aspect='auto')
                    ot_ax.set_xticks([])
                    ot_ax.set_yticks([])
                    plt.colorbar(im, ax=ot_ax, shrink=0.9)

                    im = final_ax.imshow((ot_cost_final * ot_weight + action_inconsistency_buffer[::Ta]).reshape(1, -1), cmap='magma', aspect='auto')
                    final_ax.set_xticks([])
                    final_ax.set_yticks([])
                    plt.colorbar(im, ax=final_ax, shrink=0.9)

                    action_ax.set_title('Action Inconsistency')
                    ot_ax.set_title('OT Cost')
                    final_ax.set_title('Failure index')
                    for x in range(action_inconsistency_buffer.shape[0]//Ta):
                        rollout_array = episode['side_cam'][int(x * Ta)]
                        if episode['action_mode'][int(x * Ta)] == INTV:
                            img = process_image(rollout_array, (100, 100), highlight=True)
                        else:
                            img = process_image(rollout_array, (100, 100), highlight=False)
                        img = np.array(img)
                        imagebox = OffsetImage(img, zoom=1)
                        trans = BlendedGenericTransform(final_ax.transData, final_ax.transAxes)
                        box_alignment = (0.5, 1.0)
                        ab = AnnotationBbox(imagebox, (x, -0.05), xycoords=trans, frameon=False,
                                            box_alignment=box_alignment, pad=0)
                        final_ax.add_artist(ab)

                    final_ax.set_xlim(-0.5, action_inconsistency_buffer.shape[0]//Ta-0.5)
                    final_ax.set_ylim(-0.5, 0.5)
                    plt.savefig(f'{eval_cfg.save_buffer_path}/episode_{episode_id}.png', bbox_inches='tight')
                    break

            # robot.send_joint_pose(robot.home_joint_pos)
            # time.sleep(2)
            robot.send_tcp_pose(robot.init_pose)
            time.sleep(3)
            gripper.move(gripper.max_width)
            time.sleep(0.5)
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