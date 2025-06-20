if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_hybrid_dinov2_policy import DiffusionTransformerHybridDinov2Policy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

def mkdir_p(folder_path):
    from os import makedirs, path
    from errno import EEXIST
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

class BaselineDiffusionTransformerHybridDinoMultiGPUWorkspaceSaveData(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, rank, world_size, device_id, device='cuda:0', output_dir=None):
        super().__init__(cfg, output_dir=cfg.output_dir)
        mkdir_p(cfg.output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerHybridDinov2Policy = hydra.utils.instantiate(cfg.policy).to(device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(self.model, device_ids=[device_id], find_unused_parameters=True)

        self.ema_model: DiffusionTransformerHybridDinov2Policy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

    def run(self, rank, world_size, device_id, device):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path, exclude_keys=['optimizer'], include_keys=[])

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.training.seed, drop_last=True)
        train_dataloader = DataLoader(dataset, sampler=sampler, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        self.model.module.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.module.set_normalizer(normalizer)

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # device transfer
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        full_x = []
        full_y = []

        with tqdm.tqdm(train_dataloader, desc=f"Extracting data for baseline training", disable=(rank!=0)) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                mod = self.ema_model.module # See "policy" folder
                nobs = mod.normalizer.normalize(batch['obs'])
                nactions = mod.normalizer['action'].normalize(batch['action'])
                batch_size = nactions.shape[0]

                # handle different ways of passing observation
                trajectory = nactions.reshape(batch_size, -1)
                # Get latent representation of observations
                this_nobs = dict_apply(nobs, 
                    lambda x: x[:,:mod.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                nobs_features = mod.obs_encoder.get_dense_feats(this_nobs)
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)

                print(f'At batch {batch_idx}/{len(train_dataloader)}')
                print(f'X: {global_cond.shape}, Y: {trajectory.shape}')
                full_x.append(global_cond.cpu()); full_y.append(trajectory.cpu())

        full_x = torch.cat(full_x, dim=0)
        full_y = torch.cat(full_y, dim=0)
        print(f'Full X: {full_x.shape}, Full Y: {full_y.shape}')
        torch.save({'X': full_x, 'Y': full_y}, os.path.join(cfg.output_dir, 'training_data.pt'))
                
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = BaselineDiffusionTransformerHybridDinoMultiGPUWorkspaceSaveData(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
