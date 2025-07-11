import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import BlendedGenericTransform
from PIL import Image
import torch
from robot_env import INTV

def process_image(array, size, highlight=False, color = [255, 0, 0]):
    """Process image for visualization with optional highlighting border"""
    img = Image.fromarray(array).convert('RGBA')
    img = img.resize(size)
    if highlight:
        border_width = 15
        arr = np.array(img)
        arr[:border_width, :] = color + [255]
        arr[-border_width:, :] = color + [255]
        arr[:, :border_width] = color + [255]
        arr[:, -border_width:] = color + [255]
        return Image.fromarray(arr)
    return img


def create_failure_visualization(action_inconsistency_buffer, greedy_ot_plan, greedy_ot_cost, 
                                human_eps_len, max_episode_length, Ta, episode, 
                                eps_side_img, demo_len, failure_indices, cell_size=1):
    """Create comprehensive visualization of failure detection metrics"""
    greedy_ot_cost = greedy_ot_cost[:len(action_inconsistency_buffer)//Ta] if len(greedy_ot_cost) >= len(action_inconsistency_buffer)//Ta \
        else torch.cat((greedy_ot_cost, torch.zeros((len(action_inconsistency_buffer)//Ta - len(greedy_ot_cost),), device=greedy_ot_cost.device)))
    greedy_ot_plan = greedy_ot_plan[:, :len(action_inconsistency_buffer)//Ta] if greedy_ot_plan.shape[1] >= len(action_inconsistency_buffer)//Ta \
        else torch.cat((greedy_ot_plan, torch.zeros((greedy_ot_plan.shape[0], len(action_inconsistency_buffer)//Ta - greedy_ot_plan.shape[1]), device=greedy_ot_cost.device)), 1)
    # Process OT cost and entropy
    ot_cost_final = greedy_ot_cost.detach().cpu().numpy()
    ot_entropy_final = torch.sum(-torch.log(torch.clamp(greedy_ot_plan * float(max_episode_length//Ta), min=1e-4)) * 
                              greedy_ot_plan * float(max_episode_length//Ta), dim=0).detach().cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(episode['wrist_cam'].shape[0] // Ta * cell_size, (3 + demo_len // Ta) * cell_size + 2))
    gs = plt.GridSpec(4, 1, height_ratios=[0.083, 0.083, 0.083, 0.75], hspace=0.8)
    action_ax = fig.add_subplot(gs[0])
    ot_ax = fig.add_subplot(gs[1])
    entropy_ax = fig.add_subplot(gs[2])
    plan_ax = fig.add_subplot(gs[3])

    # Action inconsistency visualization
    im = action_ax.imshow(action_inconsistency_buffer[::Ta].reshape(1, -1), cmap='plasma', aspect='auto')
    action_ax.set_xticks([])
    action_ax.set_yticks([])
    plt.colorbar(im, ax=action_ax, shrink=0.9)

    # OT cost visualization
    im = ot_ax.imshow(ot_cost_final.reshape(1, -1), cmap='cividis', aspect='auto')
    ot_ax.set_xticks([])
    ot_ax.set_yticks([])
    plt.colorbar(im, ax=ot_ax, shrink=0.9)

    # OT entropy visualization
    im = entropy_ax.imshow(ot_entropy_final.reshape(1, -1), cmap='magma', aspect='auto')
    entropy_ax.set_xticks([])
    entropy_ax.set_yticks([])
    plt.colorbar(im, ax=entropy_ax, shrink=0.9)

    # OT plan visualization
    im = plan_ax.imshow(greedy_ot_plan[:, :len(action_inconsistency_buffer)//Ta].detach().cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=plan_ax, shrink=0.75)
    plan_ax.set_xticks(np.arange(demo_len // Ta))
    plan_ax.set_yticks(np.arange(len(action_inconsistency_buffer)//Ta))
    plan_ax.set_xticklabels([''] * (demo_len // Ta))
    plan_ax.set_yticklabels([''] * (len(action_inconsistency_buffer)//Ta))
    plan_ax.tick_params(axis='both', which='both', length=0)

    # Set titles
    action_ax.set_title('Action Inconsistency', fontsize=20)
    ot_ax.set_title('OT Cost', fontsize=20)
    entropy_ax.set_title('OT entropy index', fontsize=20)
    plan_ax.set_title('OT plan', fontsize=20)
    
    # Add expert demonstration images
    for y in range(demo_len // Ta):
        human_array = (eps_side_img[int(y * Ta)].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        img = process_image(human_array, (100, 100), highlight=False)
        img = np.array(img)
        imagebox = OffsetImage(img, zoom=1)
        trans = BlendedGenericTransform(plan_ax.transAxes, plan_ax.transData)
        box_alignment = (1.0, 0.5)
        ab = AnnotationBbox(imagebox, (-0.05, y), xycoords=trans, frameon=False,
                           box_alignment=box_alignment, pad=0)
        plan_ax.add_artist(ab)

    # Add rollout trajectory images
    for x in range(action_inconsistency_buffer.shape[0]//Ta):
        rollout_array = episode['side_cam'][int(x * Ta)]
        if x in failure_indices:
            img = process_image(rollout_array, (100, 100), highlight=True, color=[0,0,255])
        elif episode['action_mode'][int(x * Ta)] == INTV:  # INTV value
            img = process_image(rollout_array, (100, 100), highlight=True)
        else:
            img = process_image(rollout_array, (100, 100), highlight=False)
        img = np.array(img)
        imagebox = OffsetImage(img, zoom=1)
        trans = BlendedGenericTransform(plan_ax.transData, plan_ax.transAxes)
        box_alignment = (0.5, 1.0)
        ab = AnnotationBbox(imagebox, (x, -0.05), xycoords=trans, frameon=False,
                           box_alignment=box_alignment, pad=0)
        plan_ax.add_artist(ab)

    # Set plot limits
    plan_ax.set_xlim(-0.5, action_inconsistency_buffer.shape[0]//Ta-0.5)
    plan_ax.set_ylim(demo_len//Ta-0.5, 0.5)
    
    return fig 