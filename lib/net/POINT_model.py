import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_model import UNet
from .FE_layer import FE_layer


class PNet(nn.Module):
    def __init__(self, n_channels, bilinear=True, patch_neighbor_size=5):
        super(PNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.patch_neighbor_size = patch_neighbor_size
        self.UNet = UNet(self.n_channels, bilinear=True)
        self.FE_layer = FE_layer(self.patch_neighbor_size)
    
    def init_network(self, device):
        self.device = device
    
    def set_input(self, input):
        self.input_drr1 = input[0].to(self.device)
        self.input_drr2 = input[1].to(self.device)
        self.correnspondence_2D = input[2].to(self.device)

    def forward(self):
        