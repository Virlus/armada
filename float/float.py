import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from collections import OrderedDict
import tqdm
from omegaconf import DictConfig

from .async_failure_detector import AsyncFailureDetectionModule
from float.util import find_matching_expert_demo, cosine_distance, optimal_transport_plan, rematch_expert_episode, OTVisualizationModule

from armada.utils.episode_manager import EpisodeManager
from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from hardware.robot_env import INTV, HUMAN


class FLOAT(AsyncFailureDetectionModule):
    """Failure detection module (FLOAT) using optimal transport matching only."""

    def __init__(self, 
                 max_queue_size: int = 3,
                 num_samples: int = 4,
                 num_expert_candidates: int = 50,
                 ot_percentile: float = 95,
                 soft_ot_ratio: float = 0.2,
                 update_stats: bool = False,
                 Ta: int = 8,
                 train_dataset_path: str = None,
                 save_buffer_path: str = None,
                 output_dir: str = None,
                 enable_visualization: bool = False,
                 To: int = 2,
                 ee_pose_dim: List = None,
                 image_shape: List = None
                 ) -> None:
        super().__init__(max_queue_size=max_queue_size)

        self.device: Optional[torch.device] = None
        self.policy: Optional[Any] = None
        self.replay_buffer: Optional[Any] = None
        self.episode_manager: Optional[Any] = None

        # Core policy params, initialized in runtime_initialize
        self.obs_feature_dim: Optional[int] = None
        self.max_episode_length: Optional[int] = None

        # Rollout onfigs
        self.num_samples: int = num_samples
        self.num_expert_candidates: int = num_expert_candidates
        self.ot_percentile: float = ot_percentile
        self.soft_ot_ratio: float = soft_ot_ratio
        self.update_stats: bool = update_stats
        self.Ta: int = Ta
        self.train_dataset_path: str = train_dataset_path
        self.save_buffer_path: str = save_buffer_path
        self.To: int = To
        self.ee_pose_dim: int = ee_pose_dim[0]
        self.img_shape: Tuple[int, int, int] = image_shape

        # Async working state, initialized in runtime_initialize
        self.all_human_latent: List[torch.Tensor] = []
        self.human_demo_indices: List[int] = []
        self.human_eps_len: List[int] = []
        self.candidate_expert_indices: List[int] = []
        self.matched_human_idx: Optional[int] = None
        self.human_latent: Optional[torch.Tensor] = None
        self.demo_len: Optional[int] = None
        self.expert_weight: Optional[torch.Tensor] = None
        self.expert_indices: Optional[torch.Tensor] = None
        self.greedy_ot_plan: Optional[torch.Tensor] = None
        self.greedy_ot_cost: Optional[torch.Tensor] = None
        self.rollout_latent: Optional[torch.Tensor] = None
        self.failure_logs: "OrderedDict[int, str]" = OrderedDict()

        # Visualization
        if enable_visualization and output_dir is not None:
            self.ot_visualizer = OTVisualizationModule(enable_visualization=True)
            self.ot_visualizer.initialize(output_dir=output_dir, fps=10)
        else:
            self.ot_visualizer = None
        self._current_robot_state: Optional[Dict[str, Any]] = None

        # Thresholds and success statistics
        self.expert_ot_threshold: Optional[float] = None
        self.success_ot_values: np.ndarray = np.zeros((0,))

    # ===================== Async handler =====================
    def handle_async_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        task_type = task.get("task_type")

        if task_type == "ot_matching":
            try:
                idx: int = task["idx"]
                rollout_latent: torch.Tensor = task["rollout_latent"]

                candidate_expert_indices: List[int] = task["candidate_expert_indices"]
                candidate_expert_latents: List[torch.Tensor] = task["candidate_expert_latents"]
                human_demo_indices: List[int] = task["human_demo_indices"]
                all_human_latent: List[torch.Tensor] = task["all_human_latent"]
                human_eps_len: List[int] = task["human_eps_len"]
                replay_buffer = task["replay_buffer"]
                device: torch.device = task["device"]
                Ta: int = task["Ta"]
                max_episode_length: int = task["max_episode_length"]

                # Rematch expert episode for better alignment
                candidate_expert_indices = rematch_expert_episode(
                    candidate_expert_latents,
                    candidate_expert_indices,
                    rollout_latent[:idx + 1]
                )

                matched_human_idx = human_demo_indices[candidate_expert_indices[0]]
                human_latent = all_human_latent[matched_human_idx]
                demo_len = human_eps_len[matched_human_idx]

                # Greedy OT matching using current rollout trajectory and the matched expert trajectory
                partial_dist_mat = torch.cat((
                    cosine_distance(human_latent, rollout_latent[:idx + 1]).to(device).detach(),
                    torch.full((demo_len // Ta, max_episode_length // Ta - idx - 1), 0, device=device)
                ), 1)

                partial_ot_plan = optimal_transport_plan(
                    human_latent,
                    torch.cat((
                        rollout_latent[:idx + 1, :],
                        torch.zeros((max_episode_length // Ta - idx - 1, rollout_latent.shape[1]), device=device)
                    ), 0),
                    partial_dist_mat
                )

                expert_weight = torch.ones((demo_len // Ta,), device=device) / float(demo_len // Ta) - torch.sum(partial_ot_plan[:, :idx + 1], dim=1)
                expert_indices = torch.nonzero(expert_weight)[:, 0]

                greedy_ot_plan = torch.cat((
                    partial_ot_plan[:, :idx + 1],
                    torch.zeros((demo_len // Ta, max_episode_length // Ta - idx - 1), device=device)
                ), 1)

                greedy_ot_cost = torch.cat((
                    torch.sum(partial_ot_plan[:, :idx + 1] * partial_dist_mat[:, :idx + 1], dim=0),
                    torch.zeros((max_episode_length // Ta - idx - 1,), device=device)
                ), 0)

                return {
                    "task_type": "ot_matching",
                    "matched_human_idx": matched_human_idx,
                    "human_latent": human_latent,
                    "demo_len": demo_len,
                    "expert_weight": expert_weight,
                    "expert_indices": expert_indices,
                    "greedy_ot_plan": greedy_ot_plan,
                    "greedy_ot_cost": greedy_ot_cost,
                    "idx": idx
                }
            except Exception as e:
                print(f"Error in ot_matching task: {e}")
                import traceback
                traceback.print_exc()
                return None

        if task_type == "failure_detection":
            try:
                greedy_ot_cost: torch.Tensor = task["greedy_ot_cost"]
                idx: int = task["idx"]
                expert_ot_threshold: Optional[float] = task["expert_ot_threshold"]

                ot_flag = False
                if expert_ot_threshold is not None:
                    ot_flag = torch.sum(greedy_ot_cost[:idx + 1]) > expert_ot_threshold

                failure_flag = ot_flag
                failure_reason = "OT violation" if ot_flag else None

                return {
                    "task_type": "failure_detection",
                    "failure_flag": failure_flag,
                    "failure_reason": failure_reason,
                    "idx": idx
                }
            except Exception as e:
                print(f"Error in failure_detection task: {e}")
                import traceback
                traceback.print_exc()
                return None

        return None

    # ===================== Public API (FailureDetectionModule) =====================
    def runtime_initialize(self, 
                           training_cfg: DictConfig,
                           device: torch.device, 
                           policy: torch.nn.Module,
                           replay_buffer: ReplayBuffer,
                           episode_manager: EpisodeManager,
                           max_episode_length: int
                           ) -> None:
        self.device = device
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.episode_manager = episode_manager
        self.obs_feature_dim = policy.obs_feature_dim
        self.max_episode_length = max_episode_length

        # Prepare human demos for OT
        self._prepare_human_demo_data()

        # Load thresholds/statistics from previous round if applicable
        self._load_success_statistics()

        # Start async thread
        self.start_async_processing()

    def detect_failure(self, **kwargs) -> Tuple[bool, Optional[str], int]:
        timestep = kwargs.get('timestep', 0)
        max_episode_length = kwargs.get('max_episode_length', self.max_episode_length)

        results = self.get_results()
        failure_flag = False
        failure_reason: Optional[str] = None
        result_idx = -1

        for result in results:
            idx = timestep // self.Ta - 1

            if result["task_type"] == "ot_matching" and result["idx"] <= idx:
                self.matched_human_idx = result["matched_human_idx"]
                self.human_latent = result["human_latent"]
                self.demo_len = result["demo_len"]
                self.expert_weight = result["expert_weight"]
                self.expert_indices = result["expert_indices"]
                self.greedy_ot_plan = result["greedy_ot_plan"]
                self.greedy_ot_cost = result["greedy_ot_cost"]

                if self.ot_visualizer is not None and result["idx"] >= 0:
                    current_ot_cost = self.greedy_ot_cost[result["idx"]].item()
                    cumulative_ot_cost = torch.sum(self.greedy_ot_cost[:result["idx"] + 1]).item()

                    side_img = None
                    if self._current_robot_state is not None:
                        if 'demo_side_img' in self._current_robot_state:
                            side_img_tensor = self._current_robot_state['demo_side_img']
                            if side_img_tensor.shape[0] == 3:
                                side_img_tensor = side_img_tensor.permute(1, 2, 0)
                            side_img = (side_img_tensor.detach().cpu().numpy()).astype(np.uint8)

                    self.ot_visualizer.add_step(
                        timestep=result["idx"],
                        ot_cost=current_ot_cost,
                        side_image=side_img,
                        cumulative_ot_cost=cumulative_ot_cost
                    )

                self.submit_task({
                    "task_type": "failure_detection",
                    "idx": result["idx"],
                    "greedy_ot_cost": self.greedy_ot_cost.clone(),
                    "expert_ot_threshold": self.expert_ot_threshold,
                    "max_episode_length": self.max_episode_length
                })

            elif result["task_type"] == "failure_detection" and result["idx"] <= idx:
                failure_flag = result["failure_flag"]
                failure_reason = result["failure_reason"]
                result_idx = max(result["idx"], result_idx)
                print(f"=========== Received failure detection result for timestep: {result_idx} =============")

                if failure_flag:
                    failure_type = "OT"
                    self.failure_logs[idx] = failure_type
                    break

        if not failure_flag and timestep >= max_episode_length - self.Ta:
            failure_reason = "maximum episode length reached"

        return failure_flag, failure_reason, result_idx

    def process_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        step_type = step_data['step_type']

        if step_type == 'episode_start':
            self.failure_logs = OrderedDict()

            if self.ot_visualizer is not None:
                episode_idx = step_data.get('episode_idx', 0)
                ot_threshold = self.expert_ot_threshold if hasattr(self, 'expert_ot_threshold') else None
                self.ot_visualizer.start_episode(episode_idx=episode_idx, ot_threshold=ot_threshold)

            rollout_init_latent = step_data['rollout_init_latent'] # TODO: add this input to the runner
            self.candidate_expert_indices = find_matching_expert_demo(
                rollout_init_latent,
                self.all_human_latent,
                self.num_expert_candidates,
                self.device
            )

            self.matched_human_idx = self.human_demo_indices[self.candidate_expert_indices[0]]
            self.human_latent = self.all_human_latent[self.matched_human_idx]
            self.demo_len = self.human_eps_len[self.matched_human_idx]

            self.expert_weight = torch.ones((self.demo_len // self.Ta,), device=self.device) / float(self.demo_len // self.Ta)
            self.expert_indices = torch.arange(self.demo_len // self.Ta, device=self.device)
            self.greedy_ot_plan = torch.zeros((self.demo_len // self.Ta, self.max_episode_length // self.Ta), device=self.device)
            self.greedy_ot_cost = torch.zeros((self.max_episode_length // self.Ta,), device=self.device)
            self.rollout_latent = torch.zeros((self.max_episode_length // self.Ta, int(self.To * self.obs_feature_dim)), device=self.device)

            return {'matched_human_idx': self.matched_human_idx}

        elif step_type == 'policy_step':
            curr_latent = step_data['curr_latent']
            timestep = step_data['timestep']

            if 'robot_state' in step_data:
                self._current_robot_state = step_data['robot_state']

            idx = timestep // self.Ta - 1

            if idx >= 0 and curr_latent is not None:
                self.rollout_latent[idx] = curr_latent[0].reshape(-1)

                candidate_expert_latents = [self.all_human_latent[i] for i in self.candidate_expert_indices]
                self.submit_task({
                    "task_type": "ot_matching",
                    "rollout_latent": self.rollout_latent.clone(),
                    "idx": idx,
                    "candidate_expert_latents": candidate_expert_latents,
                    "candidate_expert_indices": self.candidate_expert_indices,
                    "human_demo_indices": self.human_demo_indices,
                    "all_human_latent": self.all_human_latent,
                    "human_eps_len": self.human_eps_len,
                    "replay_buffer": self.replay_buffer,
                    "device": self.device,
                    "Ta": self.Ta,
                    "max_episode_length": self.max_episode_length
                })

        return {}

    def finalize_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        episode = episode_data['episode']

        self.empty_queue()
        results = self.wait_for_final_results()

        for result in results:
            if result["task_type"] == "ot_matching":
                self.matched_human_idx = result["matched_human_idx"]
                self.human_latent = result["human_latent"]
                self.demo_len = result["demo_len"]
                self.expert_weight = result["expert_weight"]
                self.expert_indices = result["expert_indices"]
                self.greedy_ot_plan = result["greedy_ot_plan"]
                self.greedy_ot_cost = result["greedy_ot_cost"]

        self.empty_result_queue()

        failure_signal = np.zeros((episode['action_mode'].shape[0] // self.Ta,), dtype=np.bool_)
        if len(self.failure_logs) > 0:
            failure_signal[list(self.failure_logs.keys())] = 1
        failure_indices = np.repeat(failure_signal, self.Ta)

        if not self.update_stats:
            return {'failure_indices': failure_indices}

        success = INTV not in episode['action_mode']
        if success:
            self.update_thresholds(
                greedy_ot_cost=self.greedy_ot_cost,
                timesteps=len(episode['action_mode']) // self.Ta
            )

            if len(self.failure_logs) > 0:  # False positive
                if "OT" in self.failure_logs.values():
                    self.update_percentile_fp()
                print("False positive trajectory! Raising percentiles...")
            print("OT threshold: ", self.expert_ot_threshold)
        else:
            has_ot_data = len(self.success_ot_values) > 0
            if len(self.failure_logs) == 0 and has_ot_data:
                self.update_percentile_fn()
                print("False negative trajectory! Lowering percentiles...")
                print("OT threshold: ", self.expert_ot_threshold)

        if self.ot_visualizer is not None:
            success_flag = INTV not in episode['action_mode']
            failure_reason = None
            if not success_flag and len(self.failure_logs) > 0:
                failure_types = list(self.failure_logs.values())
                if "OT" in failure_types:
                    failure_reason = "OT violation"
            self.ot_visualizer.end_episode(success=success_flag, failure_reason=failure_reason)

        return {'failure_indices': failure_indices}

    def should_stop_rewinding(self, j: int, episode_buffers: Dict[str, List], total_greedy_ot_cost: torch.Tensor) -> bool:
        if hasattr(self, 'soft_ot_ratio') and hasattr(self, 'greedy_ot_cost') and self.greedy_ot_cost is not None:
            soft_ot_threshold = self.soft_ot_ratio * total_greedy_ot_cost
            if torch.sum(self.greedy_ot_cost[:j // self.Ta]) < soft_ot_threshold:
                print("OT cost dropped below the soft threshold, stop rewinding.")
                return True

        if len(episode_buffers['action_mode']) > 0 and episode_buffers['action_mode'][-1] == INTV:
            print("Human intervention detected, stop rewinding.")
            return True

        return False

    def _rewind_ot_plan(self, j: int) -> None:
        assert self.expert_weight is not None and self.greedy_ot_plan is not None
        recovered_expert_weight = torch.zeros((self.demo_len // self.Ta,), device=self.device)
        recovered_expert_weight[self.expert_indices] = self.expert_weight.to(recovered_expert_weight.dtype)
        self.expert_weight = recovered_expert_weight + self.greedy_ot_plan[:, j // self.Ta - 1]
        self.expert_indices = torch.nonzero(self.expert_weight)[:, 0]
        self.greedy_ot_plan[:, j // self.Ta - 1] = 0.
        self.greedy_ot_cost[j // self.Ta - 1] = 0.

        if len(self.failure_logs) > 0:
            latest_failure_timestep, latest_failure_type = self.failure_logs.popitem()
            if latest_failure_timestep == j // self.Ta - 1:
                for timestep, failure_type in list(self.failure_logs.items()):
                    if timestep >= latest_failure_timestep - 1:
                        self.failure_logs.move_to_end(timestep)
                        _, _ = self.failure_logs.popitem()
                self.failure_logs[latest_failure_timestep - 1] = latest_failure_type
            else:
                self.failure_logs[latest_failure_timestep] = latest_failure_type

    def cleanup(self) -> None:
        self.stop_async_processing()
        success_stats = self.get_success_statistics()
        try:
            import os
            np.savez(os.path.join(self.save_buffer_path, 'success_stats.npz'), **success_stats)
            print("Saved success statistics")
        except Exception as e:
            print(f"Failed to save success statistics: {e}")

        if self.ot_visualizer is not None:
            self.ot_visualizer.cleanup()

    # ===================== Thresholds/Stats =====================
    def update_thresholds(self, greedy_ot_cost: Optional[torch.Tensor] = None, timesteps: Optional[int] = None):
        if greedy_ot_cost is not None and timesteps is not None:
            self.success_ot_values = np.concatenate((
                self.success_ot_values,
                np.sum(greedy_ot_cost[:timesteps].detach().cpu().numpy(), keepdims=True)
            ))
            self.expert_ot_threshold = np.percentile(self.success_ot_values, self.ot_percentile)
        return self.expert_ot_threshold, self.expert_ot_threshold

    def update_percentile_fp(self):
        self.ot_percentile = min(self.ot_percentile + 5, 100)
        if len(self.success_ot_values) > 0:
            self.expert_ot_threshold = np.percentile(self.success_ot_values, self.ot_percentile)
        return self.expert_ot_threshold, self.expert_ot_threshold

    def update_percentile_fn(self):
        self.ot_percentile = max(self.ot_percentile - 5, 0)
        if len(self.success_ot_values) > 0:
            self.expert_ot_threshold = np.percentile(self.success_ot_values, self.ot_percentile)
        return self.expert_ot_threshold, self.expert_ot_threshold

    def load_success_statistics(self, success_stats: Dict[str, Any]):
        self.success_ot_values = success_stats['ot_values']
        self.ot_percentile = success_stats.get('ot_percentile', self.ot_percentile)
        if len(self.success_ot_values) > 0:
            self.expert_ot_threshold = np.percentile(self.success_ot_values, self.ot_percentile)

    def get_success_statistics(self) -> Dict[str, Any]:
        return {
            'ot_values': self.success_ot_values,
            'ot_percentile': self.ot_percentile
        }

    # ===================== Internal helpers =====================
    def _prepare_human_demo_data(self) -> None:
        self.human_demo_indices = []
        for i in range(self.replay_buffer.n_episodes):
            episode_start = self.replay_buffer.episode_ends[i - 1] if i > 0 else 0
            if np.any(self.replay_buffer.data['action_mode'][episode_start: self.replay_buffer.episode_ends[i]] == HUMAN):
                self.human_demo_indices.append(i)

        self.all_human_latent = []
        self.human_eps_len = []

        from torchvision.transforms import CenterCrop
        for i in tqdm.tqdm(self.human_demo_indices, desc="Obtaining latent for human demo"):
            human_episode = self.replay_buffer.get_episode(i)
            side_img_processor = CenterCrop((self.img_shape[1], self.img_shape[2]))
            wrist_img_processor = CenterCrop((self.img_shape[1], self.img_shape[2]))

            eps_side_img = (side_img_processor(torch.from_numpy(human_episode['side_cam']).permute(0, 3, 1, 2)) / 255.0).to(self.device)
            eps_wrist_img = (wrist_img_processor(torch.from_numpy(human_episode['wrist_cam']).permute(0, 3, 1, 2)) / 255.0).to(self.device)

            if 'tcp_pose' in human_episode:
                eps_state = np.zeros((human_episode['tcp_pose'].shape[0], self.ee_pose_dim))
                eps_state[:, :3] = human_episode['tcp_pose'][:, :3]
                if hasattr(self.episode_manager, 'obs_rot_transformer') and self.episode_manager.obs_rot_transformer:
                    eps_state[:, 3:] = self.episode_manager.obs_rot_transformer.forward(human_episode['tcp_pose'][:, 3:])
                else:
                    eps_state[:, 3:] = human_episode['tcp_pose'][:, 3:]
            else:
                eps_state = human_episode['joint_pos']

            eps_state = torch.from_numpy(eps_state).to(self.device)
            demo_len = human_episode['action'].shape[0]

            human_latent = torch.zeros((self.max_episode_length // self.Ta, int(self.To * self.obs_feature_dim)), device=self.device)
            for idx in range(self.max_episode_length // self.Ta):
                human_demo_idx = min(idx * self.Ta, (demo_len // self.Ta - 1) * self.Ta)
                if human_demo_idx < self.To - 1:
                    indices = [0] * (self.To - 1 - human_demo_idx) + list(range(human_demo_idx + 1))
                    obs_dict = {
                        'side_img': eps_side_img[indices, :].unsqueeze(0),
                        'wrist_img': eps_wrist_img[indices, :].unsqueeze(0),
                        self.episode_manager.state_type: eps_state[indices, :].unsqueeze(0)
                    }
                else:
                    obs_dict = {
                        'side_img': eps_side_img[human_demo_idx - self.To + 1: human_demo_idx + 1, :].unsqueeze(0),
                        'wrist_img': eps_wrist_img[human_demo_idx - self.To + 1: human_demo_idx + 1, :].unsqueeze(0),
                        self.episode_manager.state_type: eps_state[human_demo_idx - self.To + 1: human_demo_idx + 1, :].unsqueeze(0)
                    }

                with torch.no_grad():
                    obs_features = self.policy.extract_latent(obs_dict)
                    human_latent[idx] = obs_features.squeeze(0).reshape(-1)

            self.human_eps_len.append(self.max_episode_length)
            self.all_human_latent.append(human_latent)

    def _load_success_statistics(self) -> None:
        import re
        match_round = re.search(r'round(\d)', self.train_dataset_path)
        training_set_num_round = int(match_round.group(1)) if match_round else 0

        match_round = re.search(r'round(\d)', self.save_buffer_path)
        current_round = int(match_round.group(1)) if match_round else 0

        if training_set_num_round != current_round:
            print("Re-initializing success statistics for the current round.")
        else:
            print("Loading success statistics from the previous round.")
            try:
                import os
                prev_success_states = np.load(os.path.join(self.train_dataset_path, 'success_stats.npz'))
                self.load_success_statistics(prev_success_states)
            except FileNotFoundError:
                print("No previous success statistics found, initializing fresh.") 