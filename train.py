import numpy as np
import torch.nn as nn
from tqdm import tqdm

#def run_train(args, model, device, train_loader, optimizer, epoch):
def run_train(args, model, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.BCELoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
#        data, target = data.to(device), target.to(device)
        data, target = data, target.float()
        optimizer.zero_grad()
        output = model(data.float()).flatten()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

