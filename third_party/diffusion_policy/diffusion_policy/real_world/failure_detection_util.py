import numpy as np
from typing import Union, Dict, Callable, Optional
import torch
from torch import Tensor
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

def cosine_distance(x, y):
    C = torch.mm(x, y.T)
    x_norm = torch.norm(x, p=2, dim=1)
    y_norm = torch.norm(y, p=2, dim=1)
    x_n = x_norm.unsqueeze(1)
    y_n = y_norm.unsqueeze(1)
    norms = torch.mm(x_n, y_n.T)
    C = (1 - C / norms)
    return C

def tensor_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def greedy_ot_amortize(
    greedy_ot_plan: Tensor,
    greedy_ot_cost: Tensor,
    expert_weight: Tensor, 
    rollout_weight: float, 
    dist2expert: Tensor,
    expert_indices: Tensor,
    rollout_index: int,
):
    '''
    Implementation of the greedy OT amortization algorithm.
    Args:
        greedy_ot_plan: tensor of shape (num_experts, num_rollouts) representing the current OT plan.
        greedy_ot_cost: tensor of shape (num_rollouts,) representing the current cost for each rollout.
        expert_weight: tensor of shape (num_experts,) representing the weight of each expert.
        rollout_weight: float representing the weight of the current rollout.
        dist2expert: tensor of shape (num_experts,) representing the distance from the rollout to each expert.
        expert_indices: tensor of shape (num_experts,) representing the indices of experts.
        rollout_index: int representing the index of the current rollout.
    Returns:
        greedy_ot_plan: updated tensor of shape (num_experts, num_rollouts) with the amortized OT plan.
        greedy_ot_cost: updated tensor of shape (num_rollouts,) with the amortized costs.
        expert_weight: updated tensor of shape (num_experts,) with the remaining weights of experts.
        expert_indices: updated tensor of shape (num_experts,) with the remaining indices of experts.
    '''
    while rollout_weight > 0:
        if dist2expert.shape[0] == 0:
            break
        nearest_expert = torch.argmin(dist2expert, dim=0)
        expert_index = expert_indices[nearest_expert]
        if expert_weight[nearest_expert] < rollout_weight:
            rollout_weight -= expert_weight[nearest_expert]
            greedy_ot_plan[expert_index, rollout_index] += expert_weight[nearest_expert]
            greedy_ot_cost[rollout_index] += expert_weight[nearest_expert] * dist2expert[nearest_expert]
            # Release related expert from the buffer
            expert_weight = tensor_delete(expert_weight, nearest_expert)
            expert_indices = tensor_delete(expert_indices, nearest_expert)
            dist2expert = tensor_delete(dist2expert, nearest_expert)
        else:
            greedy_ot_plan[expert_index, rollout_index] += rollout_weight
            greedy_ot_cost[rollout_index] += rollout_weight * dist2expert[nearest_expert]
            expert_weight[nearest_expert] -= rollout_weight
            rollout_weight = 0
            if expert_weight[nearest_expert] == 0:
                expert_weight = tensor_delete(expert_weight, nearest_expert)
                expert_indices = tensor_delete(expert_indices, nearest_expert)
                dist2expert = tensor_delete(dist2expert, nearest_expert)
    
    return greedy_ot_plan, greedy_ot_cost, expert_weight, expert_indices