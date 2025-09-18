from omegaconf import DictConfig
from typing import List
import torch


class BaseEnvRunner:
    def __init__(self, 
                 cfg: DictConfig,
                 rank: int,
                 device_ids: List[int]):
        self.cfg = cfg
        self.rank = rank
        self.device_ids = device_ids
        self.world_size = len(device_ids)
        self.device_id = device_ids[rank]
        self.device = f"cuda:{self.device_id}"
        self.fps = cfg.fps

        torch.distributed.init_process_group("nccl", rank=rank, world_size=self.world_size)

    def run_rollout(self):
        raise NotImplementedError()