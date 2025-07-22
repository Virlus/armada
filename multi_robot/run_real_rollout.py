#!/usr/bin/env python3
"""
Main entry point for real robot rollouts using the modular environment runner.
Use different config files to specify rollout style and failure detection approach.

Usage:
    python run_real_rollout.py --config-name real_failure_detection
    python run_real_rollout.py --config-name rollout_FAIL_DETECT  
    python run_real_rollout.py --config-name rollout_sirius
"""

import pathlib
import sys
import os
import socket
import torch
import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diffusion_policy"))

from multi_robot.nodes.robot_node import RobotNode


def setup_distributed_training(device_ids, port_start=29999):
    """Setup distributed training environment"""
    os.environ["MASTER_ADDR"] = "localhost"
    
    # Find available port
    port = port_start
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                break
        except OSError:
            port += 1
    
    os.environ["MASTER_PORT"] = f"{port}"
    print(f"Using port {port} for distributed training")
    
    return port


def main_worker(rank: int, eval_cfg: DictConfig, device_ids: list):
    """Main worker function for each process"""
    try:
        # Create and run the robot runner
        runner = RobotNode(eval_cfg, rank, device_ids)
        runner._main_thread()
    except Exception as e:
        print(f"Error in worker {rank}: {e}")
        raise


@hydra.main(config_path='diffusion_policy/diffusion_policy/config', version_base=None)
def main(cfg: DictConfig):
    """Main entry point"""
    print(f"Starting real robot rollout with config: {cfg}")
    
    # Parse device IDs
    device_ids = [int(x) for x in cfg.device_ids.split(",")]
    
    # Setup distributed training
    setup_distributed_training(device_ids)
    
    # Run rollout
    if len(device_ids) == 1:
        # Single GPU case
        main_worker(0, cfg, device_ids)
    else:
        # Multi-GPU case
        torch.multiprocessing.spawn(
            main_worker, 
            args=(cfg, device_ids), 
            nprocs=len(device_ids), 
            join=True
        )


if __name__ == "__main__":
    main() 