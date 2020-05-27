import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        