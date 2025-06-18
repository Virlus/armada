import tqdm
import torch
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../diffusion_policy"))

import CFM.net_CFM as Net
from data_loader import adjust_xshape
from argparse import ArgumentParser
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    
parser = ArgumentParser()
parser.add_argument("--exp_name", required=True, type=str)
args = parser.parse_args()

if __name__ == "__main__":
    training_data = torch.load(f'/home/yuwenye/project/human-in-the-loop/baseline_data/training/{args.exp_name}/training_data.pt')
    X, Y = training_data['X'], training_data['Y']
    X = adjust_xshape(X, 10)
    global_cond_dim = X.shape[1]; input_dim = Y.reshape(Y.shape[0], 16, -1).shape[-1]
    print(f'Current feature shape: {global_cond_dim}')
    train_data = torch.utils.data.TensorDataset(X)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    ckpt_file = f'/home/yuwenye/project/human-in-the-loop/baseline_data/outputs/{args.exp_name}/baseline.pt'
    os.makedirs(f'/home/yuwenye/project/human-in-the-loop/baseline_data/outputs/{args.exp_name}', exist_ok=True)
    
    # choice of model/method
    net = Net.get_unet(input_dim).to(device)
    EPOCHS = 200
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
    if os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        starting_epochs = ckpt['epoch']
        losses = ckpt['losses']
    else:
        starting_epochs = 0
        losses = []

    t = tqdm.trange(starting_epochs, EPOCHS)
    for i in t:
        # Save checkpoint before to avoid nan loss
        ckpt = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': i+1,
            'losses': losses
        }
        torch.save(ckpt, ckpt_file)

        net.train()
        loss_i = []
        for (x_batch, ) in tqdm.tqdm(train_loader, desc='Training Batches'):
            observation = x_batch.to(device)
            optimizer.zero_grad()
            x0, x1 = observation, torch.randn_like(observation).to(device)
            vtrue = x1 - x0
            cont_t = torch.rand(len(x1),).to(device)
            cont_t = cont_t.view(-1, *[1 for _ in range(len(observation.shape)-1)])
            xnow = x0 + cont_t * vtrue
            time_scale = 100 # In UNet, which takes discrete time steps
            vhat = net(xnow, (cont_t.view(-1)*time_scale).long())
            loss = (vhat - vtrue).pow(2).mean()
            loss.backward()
            # Terminate if loss is NaN
            if torch.isnan(loss):
                print("Loss is NaN")
                raise ValueError(f"NaN loss at epoch {i}")
            loss_i += [loss.item()]
            optimizer.step()
        losses += [sum(loss_i)/len(loss_i)]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_title("Training loss")
        ax.plot(losses)
        plt.savefig(f"/home/yuwenye/project/human-in-the-loop/baseline_data/outputs/{args.exp_name}/training_loss.png")
        plt.close('all')