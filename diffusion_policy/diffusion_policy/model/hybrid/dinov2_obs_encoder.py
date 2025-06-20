import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torchvision.transforms import Compose, Resize
from torchvision.transforms import InterpolationMode
from diffusion_policy.model.vision.dinov2_vit import DinoVisionTransformer, vit_large, vit_small, vit_base
from diffusion_policy.model.dinov2.mlp import Mlp
from collections import OrderedDict


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)


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
        
        self.wrist_img_backbone = vit_base(
            img_size=img_size,
            patch_size=patch_size,
            init_values=init_values,
            ffn_layer=ffn_layer,
            block_chunks=block_chunks,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset
        )
        
        self.side_img_backbone = vit_base(
            img_size=img_size,
            patch_size=patch_size,
            init_values=init_values,
            ffn_layer=ffn_layer,
            block_chunks=block_chunks,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset
        )
        
        self.wrist_img_head = Mlp(
            in_features=self.wrist_img_backbone.embed_dim,
            hidden_features=int(self.wrist_img_backbone.embed_dim * 0.25),
            out_features=64,
            act_layer=nn.ReLU
        )
        
        self.side_img_head = Mlp(
            in_features=self.side_img_backbone.embed_dim,
            hidden_features=int(self.side_img_backbone.embed_dim * 0.25),
            out_features=64,
            act_layer=nn.ReLU
        )
        
        self.wrist_img_head.apply(_init_weights)
        self.side_img_head.apply(_init_weights)
        
        if pretrained_path is not None:
            self.wrist_img_backbone.load_state_dict(torch.load(pretrained_path), strict=True)
            self.side_img_backbone.load_state_dict(torch.load(pretrained_path), strict=True)
            print("Loaded pretrained obs encoder from", pretrained_path)
            
        self.wrist_img_enc = nn.Sequential(self.wrist_img_backbone, self.wrist_img_head)
        self.side_img_enc = nn.Sequential(self.side_img_backbone, self.side_img_head)
        
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
    
    def get_dense_feats(self, obs):
        feats = []
        for key, encoder in self.obskey2enc.items():
            if key in obs and encoder is not None:
                obs_feat = encoder[0](obs[key])
                feats.append(obs_feat)
            elif encoder is None:
                feats.append(obs[key])
        dense_feats = torch.cat(feats, dim=-1)
        return dense_feats
    
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
        
        