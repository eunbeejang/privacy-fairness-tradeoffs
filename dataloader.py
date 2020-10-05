from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np

class BankDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./bank-data/bank-additional-full.csv',
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = BankDataset()
train_loader = DataLoader(  dataset=dataset,
                            train=True,
                            batch_size=args.batch_size,
                            shuffle=True,
                            transform=transforms.Compose(
                                [
                                    transforms.ToTensor(),
                                    transforms.Normalize((MEAN,), (STD,)),
                                ]
                            ),
                            **kwargs
                        )

test_loader = DataLoader(   dataset=dataset,
                            train=False,
                            batch_size=args.batch_size,
                            shuffle=True,
                            transform=transforms.Compose(
                                [
                                    transforms.ToTensor(),
                                    transforms.Normalize((MEAN,), (STD,)),
                                ]
                            ),
                            **kwargs
                        )    