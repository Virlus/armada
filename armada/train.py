import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
import os
import torch
import copy
from torch.multiprocessing import Process

from diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'diffusion_policy'))

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

def main(rank, cfg: OmegaConf, device_ids):
    world_size = len(device_ids)
    device_id = device_ids[rank]
    device = f"cuda:{device_id}"
    # torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, rank, world_size, device_id, device)
    workspace.run(rank, world_size, device_id, device)
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    with hydra.initialize(config_path='./config/training'):
        cfg = hydra.compose(config_name=sys.argv[1])

    device_ids = [int(x) for x in cfg.device_ids.split(",")]
    os.environ["MASTER_ADDR"] = "localhost"
    port = 30003
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
        OmegaConf.resolve(cfg)
        processes = []
        for rank in range(len(device_ids)):
            p = Process(target=main, args=(rank, cfg, device_ids))
            p.start()     
            processes.append(p)
        for p in processes:
            p.join()
