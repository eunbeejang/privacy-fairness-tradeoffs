
import torch.nn as nn
from torch import sigmoid

"""
Get mean from baseline
MEAN = 
STD = 
"""


class logistic_regression(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.l1 = nn.Linear(emb_size, int(emb_size/2))
        self.l2 = nn.Linear(int(emb_size/2), int(emb_size/4))
        self.l3 = nn.Linear(int(emb_size/4), 1)

    def forward(self, x):
        out1 = sigmoid(self.l1(x))
        out2 = sigmoid(self.l2(out1))
        y_pred = sigmoid(self.l3(out2))
        return y_pred

    def name(self):
        return "logistic_regression"

