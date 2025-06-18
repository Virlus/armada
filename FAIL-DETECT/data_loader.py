import torch
def adjust_xshape(x, in_dim):
    total_dim = x.shape[1]
    # Calculate the padding needed to make total_dim a multiple of in_dim
    remain_dim = total_dim % in_dim
    if remain_dim > 0:
        pad = in_dim - remain_dim
        total_dim += pad
        x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device)], dim=1)
    # Calculate the padding needed to make (total_dim // in_dim) a multiple of 4
    reshaped_dim = total_dim // in_dim
    if reshaped_dim % 4 != 0:
        extra_pad = (4 - (reshaped_dim % 4)) * in_dim
        x = torch.cat([x, torch.zeros(x.shape[0], extra_pad, device=x.device)], dim=1)
    return x.reshape(x.shape[0], -1, in_dim)

def get_data(type='square', adjust_shape = True, diffusion = False):
    suffix = '_diffusion' if diffusion else '_flow'
    filename = f'{type}_data{suffix}.pt'
    data = torch.load(f'../../data/outputs/{filename}')
    X, Y = data['X'], data['Y']
    in_dim_dict = {'square': 10, 'transport': 20, 'tool_hang': 20, 'can': 10}
    in_dim = in_dim_dict[type]
    if adjust_shape:
        X = adjust_xshape(X, in_dim)
    return X, Y