import numpy as np
from typing import List
import torch
from torch import Tensor
import ot
import tqdm


def euclidean_distance(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin)) ** 2, 2))
    return c


def cosine_distance(x: Tensor, y: Tensor):
    C = torch.mm(x, y.T) #（n_x, dim)，（n_y, dim) -> (n_x,n_y)
    x_norm = torch.norm(x, p=2, dim=1)
    y_norm = torch.norm(y, p=2, dim=1)
    x_n = x_norm.unsqueeze(1)
    y_n = y_norm.unsqueeze(1)
    norms = torch.mm(x_n, y_n.T)
    C = (1 - C / norms)
    return C


def tensor_delete(tensor: Tensor, indices: Tensor):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def optimal_transport_plan(
    X: Tensor,
    Y: Tensor,
    cost_matrix: Tensor,
    niter: int = 1000,
    epsilon: float = 0.1
) -> Tensor:
    X_pot = np.ones(X.shape[0]) * (1 / X.shape[0])
    Y_pot = np.ones(Y.shape[0]) * (1 / Y.shape[0])  #X_pot and Y_pot conforms to uniform distribution
    c_m = cost_matrix.data.detach().cpu().numpy()
    transport_plan = ot.sinkhorn(X_pot, Y_pot, c_m, epsilon, numItermax=niter)
    transport_plan = torch.from_numpy(transport_plan).to(X.device)
    transport_plan.requires_grad = False
    return transport_plan


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


def rematch_expert_episode(
    candidate_expert_latent: List[Tensor],
    candidate_expert_indices: Tensor,
    curr_rollout_latent: Tensor
) -> Tensor:
    """
    Given candidate expert latent variables and current rollout latent variables, 
    use OT to sort by total transport cost, getting candidate expert indices from lowest to highest cost
    """
    ot_costs = []
    for expert_latent in candidate_expert_latent:
        dist_mat = cosine_distance(expert_latent, curr_rollout_latent) # Cost matrix
        ot_plan = optimal_transport_plan(expert_latent, curr_rollout_latent, dist_mat) # Sinkhorn algorithm to solve transport plan
        ot_cost = torch.sum(ot_plan * dist_mat) # Total transport cost OT_cost, element-wise product of cost matrix and transport plan
        ot_costs.append(ot_cost)

    candidate_expert_indices = candidate_expert_indices[Tensor(ot_costs).argsort()]  # candidate_expert_indices sorted by ot_costs from lowest to highest
    return candidate_expert_indices