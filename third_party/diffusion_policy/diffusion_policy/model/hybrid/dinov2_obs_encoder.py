import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize
from torchvision.transforms import InterpolationMode
from diffusion_policy.model.vision.dinov2_vit import DinoVisionTransformer, vit_large, vit_small, vit_base
from collections import OrderedDict


class Dinov2ObsEncoder(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        proprio_shape=[9],
        pretrained_path=None,
        ):
        super(Dinov2ObsEncoder, self).__init__()
        
        self.wrist_img_enc = vit_base(
            img_size=img_size,
            patch_size=patch_size,
            init_values=init_values,
            ffn_layer=ffn_layer,
            block_chunks=block_chunks,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset
        )
        
        self.side_img_enc = vit_base(
            img_size=img_size,
            patch_size=patch_size,
            init_values=init_values,
            ffn_layer=ffn_layer,
            block_chunks=block_chunks,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset
        )
        
        if pretrained_path is not None:
            self.wrist_img_enc.load_state_dict(torch.load(pretrained_path), strict=True)
            self.side_img_enc.load_state_dict(torch.load(pretrained_path), strict=True)
            print("Loaded pretrained obs encoder from", pretrained_path)
        
        self.proprio_enc = None
        
        self.img_size = img_size
        self.proprio_size = proprio_shape[0]
        
        self.obskey2enc = OrderedDict({
            'wrist_img': self.wrist_img_enc,
            'side_img': self.side_img_enc,
            'ee_pose': self.proprio_enc
        })
    
    def forward(self, obs):
        feats = []
        for key, encoder in self.obskey2enc.items():
            if key in obs and encoder is not None:
                obs_feat = encoder(obs[key])
                feats.append(obs_feat)
            elif encoder is None:
                feats.append(obs[key])
        encoder_output = torch.cat(feats, dim=-1)
        return encoder_output
    
    def output_shape(self):
        feat_dim = 0
        for key, encoder in self.obskey2enc.items():
            if 'img' in key:
                obs = torch.randn(1, 3, self.img_size, self.img_size)
                out = encoder(obs)
                feat_dim += int(torch.prod(torch.tensor(out.shape[1:])))
            else:
                obs = torch.randn(1, self.proprio_size)
                out = encoder(obs) if encoder is not None else obs
                feat_dim += int(torch.prod(torch.tensor(out.shape[1:])))
        return [feat_dim]
        
        