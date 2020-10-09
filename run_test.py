import torch
import torch.nn as nn
from tqdm import tqdm


def test(args, model, device, test_loader, test_size):
    model.eval()
    criterion = nn.BCELoss()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for cats, conts, target in tqdm(test_loader):
            cats, conts, target = cats.to(device), conts.to(device), target.to(device)
            output = model(cats, conts)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = (output > 0.5).float()
            correct += pred.eq(target.view_as(pred)).sum().item()



    test_loss /= test_size

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            test_size,
            100.0 * correct / test_size,
        )
    )
    return 100.0 * correct / test_size

