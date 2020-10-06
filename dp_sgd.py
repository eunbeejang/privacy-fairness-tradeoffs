
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
        self.linear = nn.Linear(emb_size, 1)

    def forward(self, x):
        y_pred = sigmoid(self.linear(x))
        return y_pred

    def name(self):
        return "logistic_regression"

