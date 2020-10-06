import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

#def run_test(args, model, device, test_loader):
def run_test(args, model, test_loader, dataset_size, train_size, test_size):
    model.eval()
    criterion = nn.BCELoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
#            data, target = data.to(device), target.to(device)
            data, target = data, target.float().view(-1,1)
            output = model(data.float())
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            print(output)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print("** ", pred.eq(target.view_as(pred)).sum().item())

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

