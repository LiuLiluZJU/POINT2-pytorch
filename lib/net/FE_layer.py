""" Feature extraction layer """

import torch
import torch.nn as nn
import torch.nn.functional as F


class FE_layer(nn.module):

    def __init__(self, POI, feature_map, neighbor_size=5):
        super().__init__()
        self.POI = POI
        self.feature_map = feature_map
        self.neighbor_size = neighbor_size

    def forward():
        
