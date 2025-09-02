import torch
import numpy as np
import threading
import queue
import time
from typing import List, Dict, Tuple, Optional, Any
from scipy.spatial.transform import Rotation as R

from util.failure_detection_util import (
    cosine_distance, 
    optimal_transport_plan, 
    rematch_expert_episode
)
from util.episode_utils import EpisodeManager

class FailureDetector:
    def __init__(
        self, 
        Ta: int,
        action_inconsistency_percentile: float = 95.0,
        ot_percentile: float = 95.0,
        max_queue_size: int = 2,
        episode_manager: EpisodeManager = None,
        enable_action_inconsistency: bool = True,
        enable_OT: bool = True
    ):
        self.Ta = Ta
        self.action_inconsistency_percentile = action_inconsistency_percentile
        self.ot_percentile = ot_percentile
        self.enable_action_inconsistency = enable_action_inconsistency
        self.enable_OT = enable_OT
        # Initialize from episode manager
        self.action_rot_transformer = episode_manager.action_rot_transformer
        self.action_dim = episode_manager.action_dim
        self.Ta = episode_manager.Ta
        # Initialize thresholds
        self.expert_action_threshold = None
        self.expert_ot_threshold = None
        
        # Initialize success statistics
        self.success_action_inconsistencies = []
        self.success_ot_values = np.zeros((0,))
        
        # Initialize async processing components
        self.async_queue = queue.Queue(maxsize=max_queue_size)
        self.async_result_queue = queue.Queue()
        self.async_thread_stop = threading.Event()
        self.async_thread = None

        # Initialize last predicted abs actions
        self.last_predicted_abs_actions = None
        
    def start_async_processing(self):
        """Start the asynchronous processing thread"""
        self.async_thread = threading.Thread(target=self._async_processing_thread, daemon=True)
        self.async_thread.start()
        
    def stop_async_processing(self):
        """Stop the asynchronous processing thread"""
        if self.async_thread is not None:
            self.async_thread_stop.set()
            self.async_thread.join(timeout=1.0) # Wait for async thread to finish, max 1 second
            self.async_thread = None
    
    def _async_processing_thread(self):
        """Thread function for asynchronous processing"""
        # ot_matching and action_inconsistency are basic computation tasks, failure_detection is decision task
        while not self.async_thread_stop.is_set():
            try:
                # Get data with a timeout to allow checking the stop flag
                data = self.async_queue.get(timeout=0.1)
                
                # Unpack the data
                task_type = data["task_type"]
                
                if task_type == "ot_matching":
                    # OT Matching task
                    idx = data["idx"]
                    rollout_latent = data["rollout_latent"]
                    
                    # Rematch an expert demonstration for better alignment
                    candidate_expert_indices = rematch_expert_episode(     # OT algorithm to get candidate experts
                        data["candidate_expert_latents"], 
                        data["candidate_expert_indices"], 
                        rollout_latent[:idx+1]
                    )
                    
                    matched_human_idx = data["human_demo_indices"][candidate_expert_indices[0]]
                    human_latent = data["all_human_latent"][matched_human_idx]
                    demo_len = data["human_eps_len"][matched_human_idx]
                    human_episode = data["replay_buffer"].get_episode(matched_human_idx)
                    eps_side_img = (torch.from_numpy(human_episode['side_cam']).permute(0, 3, 1, 2) / 255.0).to(data["device"])
                    
                    # Compute OT plan and cost
                    Ta = data["Ta"]
                    max_episode_length = data["max_episode_length"]
                    
                    partial_dist_mat = torch.cat((
                        cosine_distance(human_latent, rollout_latent[:idx+1]).to(data["device"]).detach(),  # (n_x, dim), (n_y, dim) -> (n_x,n_y)
                        torch.full((demo_len // Ta, max_episode_length // Ta - idx - 1), 0, device=data["device"])
                    ), 1)  # Construct a "partial" cost matrix, only compute cosine distance between current generated trajectory (rollout_latent[:idx+1]) and expert trajectory, ungenerated part filled with 0

                    partial_ot_plan = optimal_transport_plan(
                        human_latent, 
                        torch.cat((
                            rollout_latent[:idx+1, :], 
                            torch.zeros((max_episode_length // Ta - idx - 1, rollout_latent.shape[1]), device=data["device"])
                        ), 0), 
                        partial_dist_mat
                    )  # Partial optimal transport plan partial_ot_plan
                    
                    expert_weight = torch.ones((demo_len // Ta,), device=data["device"]) / float(demo_len // Ta) - torch.sum(partial_ot_plan[:, :idx+1], dim=1)  # Initial expert weight is uniform distribution 1/(demo_len//Ta), minus transport mass already assigned to current trajectory sum(partial_ot_plan[:, :idx+1], dim=1)
                    expert_indices = torch.nonzero(expert_weight)[:, 0]  # Remaining weight expert_weight represents unmatched parts of expert samples, non-zero indices expert_indices are insufficiently matched expert samples
                    
                    greedy_ot_plan = torch.cat((
                        partial_ot_plan[:, :idx+1], 
                        torch.zeros((demo_len // Ta, max_episode_length // Ta - idx - 1), device=data["device"])
                    ), 1)
                    
                    greedy_ot_cost = torch.cat((
                        torch.sum(partial_ot_plan[:, :idx+1] * partial_dist_mat[:, :idx+1], dim=0), 
                        torch.zeros((max_episode_length // Ta - idx - 1,), device=data["device"])
                    ), 0)

                elif task_type == "action_inconsistency": # Calculate difference between actions separated by Ta steps
                    action_seq = data["action_seq"]
                    predicted_abs_actions = data["predicted_abs_actions"]
                    idx = data["idx"]
                    last_p = data["last_p"]
                    last_r = data["last_r"]
                    Ta = data["Ta"]
                    action_rot_transformer = data["action_rot_transformer"]
                    action_dim = data["action_dim"]
                    # Calculate action inconsistency
                    tmp_last_p = last_p
                    tmp_last_r = last_r
                    for step in range(Ta, action_seq.shape[1]): # Ignore first Ta steps of initialization actions
                        curr_p_action = action_seq[:, step, :3]
                        curr_p = tmp_last_p + curr_p_action
                        curr_r_action = action_rot_transformer.inverse(action_seq[:, step, 3:action_dim-1])
                        action_rot = R.from_quat(curr_r_action, scalar_first=True)
                        curr_r = tmp_last_r * action_rot
                        predicted_abs_actions[:, step] = np.concatenate((curr_p, curr_r.as_quat(scalar_first=True), action_seq[:, step, -1:]), -1)
                        tmp_last_p = curr_p
                        tmp_last_r = curr_r
                    if self.last_predicted_abs_actions is None:
                        self.last_predicted_abs_actions = np.concatenate((np.zeros((Ta, 8)), predicted_abs_actions[0, :-Ta]), 0) # Prevent anomalous value in the beginning
                    action_inconsistency = np.mean(np.linalg.norm(predicted_abs_actions[:, :-Ta] - self.last_predicted_abs_actions[np.newaxis, Ta:], axis=-1))
                    self.last_predicted_abs_actions = predicted_abs_actions[0]
                    result = {
                        "task_type": "action_inconsistency",
                        "action_inconsistency": action_inconsistency,
                        "idx": idx
                    }
                    self.async_result_queue.put(result)
                    continue
                    
                elif task_type == "failure_detection":
                    # Check if action discontinuity is too large, or OT cost exceeds limit, then store results
                    # Failure detection task
                    action_inconsistency_buffer = data["action_inconsistency_buffer"]
                    expert_action_threshold = data["expert_action_threshold"]
                    enable_OT = data["enable_OT"]
                    greedy_ot_cost = data["greedy_ot_cost"]
                    greedy_ot_plan = data["greedy_ot_plan"]
                    idx = data["idx"]
                    expert_ot_threshold = data["expert_ot_threshold"]
                    enable_action_inconsistency = data["enable_action_inconsistency"]
                    enable_OT = data["enable_OT"]
                    # Perform failure detection based on enabled detection types
                    inconsistency_violation = False
                    if enable_action_inconsistency and expert_action_threshold is not None:
                        inconsistency_violation = np.array(action_inconsistency_buffer).sum() > expert_action_threshold
                    
                    # OT detection is always enabled
                    ot_flag = False
                    if enable_OT and expert_ot_threshold is not None:
                        ot_flag = torch.sum(greedy_ot_cost[:idx+1]) > expert_ot_threshold
                    
                    failure_flag = inconsistency_violation or ot_flag # True indicates problem detected
                    failure_reason = "action inconsistency" if inconsistency_violation else "OT violation" if ot_flag else None
                    
                    result = {
                        "task_type": "failure_detection",
                        "failure_flag": failure_flag,
                        "failure_reason": failure_reason,
                        "idx": idx
                    }
                    self.async_result_queue.put(result)
                    continue
                
                # Put OT matching results in the result queue
                result = {
                    "task_type": "ot_matching",
                    "matched_human_idx": matched_human_idx,
                    "human_latent": human_latent,
                    "demo_len": demo_len,
                    "eps_side_img": eps_side_img,
                    "expert_weight": expert_weight,
                    "expert_indices": expert_indices,
                    "greedy_ot_plan": greedy_ot_plan,
                    "greedy_ot_cost": greedy_ot_cost,
                    "idx": idx
                }
                self.async_result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in async processing thread: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def submit_ot_matching_task(self, rollout_latent, idx, candidate_expert_latents, candidate_expert_indices, 
                               human_demo_indices, all_human_latent, human_eps_len, replay_buffer, 
                               device, max_episode_length):
        """Submit an OT matching task for asynchronous processing"""
        try:
            self.async_queue.put_nowait({
                "task_type": "ot_matching",
                "idx": idx,
                "rollout_latent": rollout_latent.clone(),
                "candidate_expert_latents": candidate_expert_latents,
                "candidate_expert_indices": candidate_expert_indices,
                "human_demo_indices": human_demo_indices,
                "all_human_latent": all_human_latent,
                "human_eps_len": human_eps_len,
                "replay_buffer": replay_buffer,
                "device": device,
                "Ta": self.Ta,
                "max_episode_length": max_episode_length
            })
            return True
        except queue.Full:
            return False
        
    def submit_action_inconsistency_task(self, action_seq, predicted_abs_actions, idx, last_p, last_r):
        """Submit an action inconsistency task for asynchronous processing"""
        try:
            self.async_queue.put_nowait({
                "task_type": "action_inconsistency",
                "action_seq": action_seq,
                "predicted_abs_actions": predicted_abs_actions,
                "action_rot_transformer": self.action_rot_transformer,
                "action_dim": self.action_dim,
                "Ta": self.Ta,
                "idx": idx,
                "last_p": last_p,
                "last_r": last_r
            })
            return True
        except queue.Full:
            return False
    
    def submit_failure_detection_task(self, action_inconsistency_buffer, idx, greedy_ot_cost, greedy_ot_plan, max_episode_length):
        """Submit a failure detection task for asynchronous processing"""
        try:
            self.async_queue.put_nowait({
                "task_type": "failure_detection",
                "action_inconsistency_buffer": action_inconsistency_buffer[:int(idx+1)*self.Ta].copy(),
                "expert_action_threshold": self.expert_action_threshold,
                "greedy_ot_cost": greedy_ot_cost.clone(),
                "greedy_ot_plan": greedy_ot_plan.clone(),
                "idx": idx,
                "expert_ot_threshold": self.expert_ot_threshold,
                "enable_action_inconsistency": self.enable_action_inconsistency,
                "enable_OT": self.enable_OT,
                "max_episode_length": max_episode_length
            })
            return True
        except queue.Full:
            return False
    
    def get_results(self) -> List[Dict]:
        """Get all available results from the result queue"""
        results = []
        try:
            while not self.async_result_queue.empty():
                results.append(self.async_result_queue.get_nowait())
        except queue.Empty:
            pass
        return results
    
    def wait_for_final_results(self, timeout=0.5) -> None:
        """Wait for final results to be processed"""
        start_wait = time.time()
        results = []
        while time.time() - start_wait < timeout:
            try:
                while not self.async_result_queue.empty():
                    result = self.async_result_queue.get_nowait()
                    if result["task_type"] == "ot_matching" or result["task_type"] == "action_inconsistency":
                        results.append(result)
            except queue.Empty:
                time.sleep(0.01)
                continue
            break
        return results
    
    def empty_queue(self) -> None:
        """Empty both the task and result queues"""
        while not self.async_queue.empty():
            try:
                self.async_queue.get_nowait()
            except queue.Empty:
                break

    def empty_result_queue(self) -> None:
        while not self.async_result_queue.empty():
            try:
                self.async_result_queue.get_nowait()
            except queue.Empty:
                break
    
    def update_thresholds(self, success_action_inconsistency=None, greedy_ot_cost=None, timesteps=None):
        """Update failure detection thresholds based on successful episodes"""
        if success_action_inconsistency is not None and self.enable_action_inconsistency:
            self.success_action_inconsistencies.append(success_action_inconsistency)
            self.expert_action_threshold = np.percentile(
                np.array(self.success_action_inconsistencies), 
                self.action_inconsistency_percentile
            )
            
        if greedy_ot_cost is not None and timesteps is not None and self.enable_OT:
            self.success_ot_values = np.concatenate((
                self.success_ot_values, 
                np.sum(greedy_ot_cost[:timesteps].detach().cpu().numpy(), keepdims=True)
            ))
            
            self.expert_ot_threshold = np.percentile(self.success_ot_values, self.ot_percentile)
                
        return self.expert_action_threshold, self.expert_ot_threshold
    
    def update_percentile_fp(self, ot_fp=False, action_fp=False):
        if ot_fp and self.enable_OT:
            self.ot_percentile = min(self.ot_percentile + 2.5, 100)
        if action_fp and self.enable_action_inconsistency:
            self.action_inconsistency_percentile = min(self.action_inconsistency_percentile + 2.5, 100)
        
        if self.enable_action_inconsistency and len(self.success_action_inconsistencies) > 0:
            self.expert_action_threshold = np.percentile(
                np.array(self.success_action_inconsistencies), 
                self.action_inconsistency_percentile
            )
        
        if self.enable_OT and len(self.success_ot_values) > 0:
            self.expert_ot_threshold = np.percentile(self.success_ot_values, self.ot_percentile)
        
        return self.expert_action_threshold, self.expert_ot_threshold
    
    def update_percentile_fn(self):
        if self.enable_OT:
            self.ot_percentile = max(self.ot_percentile - 2.5, 0)
        if self.enable_action_inconsistency:
            self.action_inconsistency_percentile = max(self.action_inconsistency_percentile - 2.5, 0)
        
        if self.enable_action_inconsistency and len(self.success_action_inconsistencies) > 0:
            self.expert_action_threshold = np.percentile(
                np.array(self.success_action_inconsistencies), 
                self.action_inconsistency_percentile
            )
        
        if self.enable_OT and len(self.success_ot_values) > 0:
            self.expert_ot_threshold = np.percentile(self.success_ot_values, self.ot_percentile)
        
        return self.expert_action_threshold, self.expert_ot_threshold
    
    def load_success_statistics(self, success_stats):
        """Load success statistics from a saved file"""
        if self.enable_action_inconsistency:
            self.success_action_inconsistencies = list(success_stats['action_inconsistencies'])
            self.action_inconsistency_percentile = success_stats['action_inconsistency_percentile'] if 'action_inconsistency_percentile' in success_stats else self.action_inconsistency_percentile
            if len(self.success_action_inconsistencies) > 0:
                self.expert_action_threshold = np.percentile(self.success_action_inconsistencies, self.action_inconsistency_percentile)
        
        if self.enable_OT:
            self.success_ot_values = success_stats['ot_values']
            self.ot_percentile = success_stats['ot_percentile'] if 'ot_percentile' in success_stats else self.ot_percentile
            if len(self.success_ot_values) > 0:
                self.expert_ot_threshold = np.percentile(self.success_ot_values, self.ot_percentile)
        
    def get_success_statistics(self):
        """Get the current success statistics for saving"""
        return {
            'action_inconsistencies': np.array(self.success_action_inconsistencies),
            'ot_values': self.success_ot_values,
            'action_inconsistency_percentile': self.action_inconsistency_percentile,
            'ot_percentile': self.ot_percentile
        }
