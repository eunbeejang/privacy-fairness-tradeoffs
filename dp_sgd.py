
import torch.nn as nn


"""
Get mean from baseline
MEAN = 
STD = 
"""


class logistic_regression(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = sigmoid(self.linear(x))
        return y_pred

    def name(self):
        return "logistic_regression"

