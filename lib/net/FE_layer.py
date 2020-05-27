""" Feature extraction layer """

import torch
import torch.nn as nn
import torch.nn.functional as F


class FE_layer(nn.Module):

    def __init__(self, patch_neighbor_size=5):
        super(FE_layer, self).__init__()
        self.patch_neighbor_size = patch_neighbor_size
        self.padding = nn.ReplicationPad2d(patch_neighbor_size)

    def forward(self, feature_map, single_POI):
        feature_map = self.padding(feature_map)
        # print("size:", feature_map.shape)
        print(single_POI[0] + 2 * self.patch_neighbor_size)
        feature_kernel = feature_map[:, :, single_POI[0] : single_POI[0] + 2 * self.patch_neighbor_size + 1, \
            single_POI[1] : single_POI[1] + 2 * self.patch_neighbor_size + 1]
        feature_kernel = feature_kernel.clone()
        # print("size:", feature_kernel.shape)

        return feature_kernel

