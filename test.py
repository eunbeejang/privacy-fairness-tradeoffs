import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

#def run_test(args, model, device, test_loader):
def run_test(args, model, test_loader, test_size):
    model.eval()
    criterion = nn.BCELoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for cats, conts, target in tqdm(test_loader):
#            data, target = data.to(device), target.to(device)
            cats, conts, target = cats, conts, target.float().view(-1,1)
            output = model(cats, conts.float())
            test_loss += criterion(output, target).item()  # sum up batch loss
            correct += output.eq(target.view_as(output)).sum().item()
            #print("** ", pred.eq(target.view_as(pred)).sum().item())

    test_loss /= test_size

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n".format(
            test_loss,
            correct,
            test_size,
            100.0 * correct / test_size,
        )
    )
    return 100.0 * correct / test_size

