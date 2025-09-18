import pathlib
import sys
import os
import socket
import torch
import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diffusion_policy"))


def setup_distributed_env(device_ids, port_start=29999):
    """Setup distributed rollout environment"""
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
    print(f"Using port {port} for distributed rollout")
    
    return port


def main_worker(rank: int, cfg: DictConfig, device_ids: list):
    """Main worker function for each process"""
    try:
        # Create and run the env runner
        cls = hydra.utils.get_class(cfg._target_)
        runner = cls(cfg, rank, device_ids)
        runner.run_rollout()
    except Exception as e:
        print(f"Error in worker {rank}: {e}")
        raise


@hydra.main(config_path='./config', version_base=None)
def main(cfg: DictConfig):
    """Main entry point"""
    print(f"Starting real robot rollout with config: {cfg}")
    
    # Parse device IDs
    device_ids = [int(x) for x in cfg.device_ids.split(",")]
    
    # Setup distributed rollout
    setup_distributed_env(device_ids)
    
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