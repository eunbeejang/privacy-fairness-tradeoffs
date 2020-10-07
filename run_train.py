import numpy as np
import torch.nn as nn
from tqdm import tqdm

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.BCELoss()
    losses = []
    for _batch_idx, (cats, conts, target) in enumerate(tqdm(train_loader)):
        cats, conts, target = cats.to(device), conts.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(cats, conts).view(-1)
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
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.10f}")
