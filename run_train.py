import numpy as np
import torch
import pytorch_lightning as pl
from torch import autograd
from tqdm import tqdm
np.random.seed(0)
torch.manual_seed(0)



def train(args, model, device, train_loader,criterion, optimizer, epoch, sigma):

    model.train()

    losses = []
    for _batch_idx, (cats, conts, target) in enumerate(tqdm(train_loader)):


        cats, conts, target = cats.to(device), conts.to(device), target.to(device)


        optimizer.zero_grad()

        output = model(cats, conts).view(-1)

        # Terminate when the model predicts NaN
        if torch.isnan(cats).any() or torch.isnan(output).any():
            #print(_batch_idx, cats, output)
            exit("ERROR: Predicted NaN")

        else:
            loss = criterion(output, target)
#            loss = model.BCELoss(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    if not args.disable_dp:
        if sigma > 0:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
            print(
                f"Train Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
            )
        else:
            print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.10f}")

    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.10f}")
