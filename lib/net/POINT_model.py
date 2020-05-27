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
        # self.loss = nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss()
        # self.sigmoid = nn.sigmoid()
        self.optimizer = torch.optim.SGD(self.UNet.parameters(), 0.001, weight_decay=1e-8, momentum=0.9)
    
    def set_input(self, input1, input2, correspondence_2D):
        self.input_drr1 = input1
        self.input_drr2 = input2
        self.correspondence_2D = correspondence_2D

    def generate_score_map_gt(self, single_POI, size_H, size_W):

        score_map_gt = torch.zeros((size_H, size_W)).cuda()
        up_bound = single_POI[2] + self.patch_neighbor_size
        low_bound = single_POI[2] - self.patch_neighbor_size
        right_bound = single_POI[3] + self.patch_neighbor_size
        left_bound = single_POI[3] - self.patch_neighbor_size

        if up_bound >= size_H:
            up_bound = size_H - 1
        if low_bound < 0:
            low_bound = 0
        if right_bound >= size_W:
            right_bound = size_W - 1
        if left_bound < 0:
            left_bound = 0
        
        score_map_gt[low_bound : up_bound + 1, left_bound : right_bound + 1] = 1
        score_map_gt = score_map_gt.unsqueeze(0).unsqueeze(0)

        return score_map_gt
        
    def forward(self):
        self.feature_map1 = self.UNet(self.input_drr1)
        self.feature_map2 = self.UNet(self.input_drr2)
        self.i_size_H = self.input_drr1.shape[2]
        self.i_size_W = self.input_drr1.shape[3]
        self.f_size_H = self.feature_map1.shape[2]
        self.f_size_W = self.feature_map1.shape[3]
        self.factor_H = self.i_size_H / self.f_size_H
        self.factor_W = self.i_size_W / self.f_size_W
    
    def backward_basic(self):
        self.loss_total = 0
        batch_size = self.correspondence_2D.shape[0]
        for batch_index in range(self.correspondence_2D.shape[0]):
            loss_batch = 0
            point_count = 0
            for point_index in range(self.correspondence_2D.shape[2]):

                if self.correspondence_2D[batch_index][0][point_index][4] == -1:
                    continue

                feature_map1_divided = self.feature_map1[batch_index].unsqueeze(0)
                drr_POI = self.correspondence_2D[batch_index][0][point_index][0 : 4].clone()
                drr_POI = torch.floor(drr_POI / self.factor_H).int()  # No need to multiply factor
                # print("self.factor_H", self.factor_H)
                # print("drr_POI:", drr_POI)
                # print("shape:", feature_map1_divided.shape)
                feature_kernel = self.FE_layer(feature_map1_divided, drr_POI)
                # print("shape:", feature_kernel.shape)
                feature_map2_divided = self.feature_map2[batch_index].unsqueeze(0)
                score_map = F.conv2d(feature_map2_divided, feature_kernel, padding=self.patch_neighbor_size)
                # print("score_map size:", score_map.shape)
                # score_map = self.sigmoid(score_map)

                score_map_gt = self.generate_score_map_gt(drr_POI, self.f_size_H, self.f_size_W)

                loss_batch += self.criterion(score_map, score_map_gt)
                point_count += 1
            
            loss_batch = loss_batch / point_count
            self.loss_total += loss_batch
        
        self.loss_total = self.loss_total / batch_size
        print("loss:", self.loss_total)
        self.loss_total.backward()
    
    def optimize_parameters(self):
        # forward
        self()
        self.optimizer.zero_grad()
        self.backward_basic()
        nn.utils.clip_grad_value_(self.UNet.parameters(), 0.1)
        self.optimizer.step()