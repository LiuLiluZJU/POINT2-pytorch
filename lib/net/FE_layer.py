""" Feature extraction layer """

import torch
import torch.nn as nn
import torch.nn.functional as F


class FE_layer(nn.module):

    def __init__(self, patch_neighbor_size=5):
        super(FE_layer, self).__init__()
        self.patch_neighbor_size = patch_neighbor_size
        self.padding = nn.ReplicationPad2d(patch_neighbor_size)

    def forward(self, feature_map, single_POI):
        feature_map = self.padding(feature_map)
        feature_kernel = feature_map[:][:][single_POI[0] + self.patch_neighbor_size][single_POI[1] + self.patch_neighbor_size]
        feature_kernel = feature_kernel.copy()

        return feature_kernel

