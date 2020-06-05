import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_model import UNet
from .FE_layer import FE_layer
import matplotlib.pyplot as plt


class PNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=64, bilinear=True, patch_neighbor_size=30):
        super(PNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_neighbor_size = patch_neighbor_size
        self.UNet = UNet(self.n_channels, self.n_classes, bilinear=True)
        self.FE_layer = FE_layer(patch_neighbor_size)
        self.padding = nn.ZeroPad2d(patch_neighbor_size)
        # self.loss = nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.batchnorm = nn.BatchNorm2d(1)
        self.kernel_weight = nn.Parameter(torch.ones((1, 64, 2 * patch_neighbor_size + 1, 2 * patch_neighbor_size + 1)))
        self.optimizer = torch.optim.Adam([
            {'params': self.UNet.parameters()},
            {'params': self.kernel_weight}
            ], lr=0.001, weight_decay=1e-8)
    
    def set_input(self, input1, input2, correspondence_2D):
        self.input_drr1 = input1
        self.input_drr2 = input2
        self.correspondence_2D = correspondence_2D
        self.batch_size = self.correspondence_2D.shape[0]
        self.point_num = self.correspondence_2D.shape[2]

    def generate_score_map_gt(self, single_POI, size_H, size_W):

        score_map_gt = torch.ones((size_H, size_W)).cuda()
        score_map_gt = score_map_gt * (-10)
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
        
        score_map_gt[low_bound : up_bound + 1, left_bound : right_bound + 1] = 10
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
        print("self.eval", self.training)
        
        if not self.training:
            # score_map_b_list = []
            for batch_index in range(self.batch_size):
                score_map_c_list = []
                max_index_c_list = []
                for point_index in range(self.point_num):
                    if self.correspondence_2D[batch_index][0][point_index][4] != -1:
                        feature_map1_divided = self.feature_map1[batch_index].unsqueeze(0)
                        drr_POI = self.correspondence_2D[batch_index][0][point_index][0 : 4].clone()
                        drr_POI[0] = torch.floor(drr_POI[0] / self.factor_H)  # No need to multiply factor
                        drr_POI[1] = torch.floor(drr_POI[1] / self.factor_W)  # No need to multiply factor
                        drr_POI[2] = torch.floor(drr_POI[2] / self.factor_H)  # No need to multiply factor
                        drr_POI[3] = torch.floor(drr_POI[3] / self.factor_W)  # No need to multiply factor
                        drr_POI = drr_POI.int()
                        feature_kernel = self.FE_layer(feature_map1_divided, drr_POI)
                        feature_map2_divided = self.feature_map2[batch_index].unsqueeze(0)
                        score_map = F.conv2d(feature_map2_divided, feature_kernel)
                        score_map = self.padding(score_map)
                        # print("score_map size:", score_map.shape)
                        # score_map_c_list.append(score_map.squeeze(1))

                        # Find index of max value
                        score_map_squeezed = torch.squeeze(score_map)
                        score_map_flattened = score_map_squeezed.view(-1)
                        max_index_flattened = score_map_flattened.argmax(dim=0)
                        max_index = torch.Tensor([max_index_flattened / self.f_size_H, max_index_flattened % self.f_size_W]).cuda()
                        max_index = max_index.int()
                        # print("equal?", score_map_squeezed[max_index[0], max_index[1]], score_map_flattened[max_index_flattened])
                        max_index_c_list.append(max_index)
                        print("eular distance:", (max_index[0] - drr_POI[2]) ** 2 + (max_index[1] - drr_POI[3]) ** 2)
                        print("original eular distance:", (drr_POI[0] - drr_POI[2]) ** 2 + (drr_POI[1] - drr_POI[3]) ** 2)
                        # print("max_index:", max_index)

                        # Show
                        # score_map_squeezed_show = score_map_squeezed.cpu().data.numpy()
                        # max_index_show = max_index.cpu().data.numpy()
                        # drr_POI_show = drr_POI.cpu().data.numpy()
                        # plt.imshow(score_map_squeezed_show, cmap='gray')
                        # plt.scatter([drr_POI_show[1]], [drr_POI_show[0]], marker='o')
                        # plt.scatter([drr_POI_show[3]], [drr_POI_show[2]], marker='+')
                        # plt.scatter([max_index_show[1]], [max_index_show[0]], marker='x')
                        # plt.show()
                    else:
                        max_index = torch.Tensor([0, 0]).int().cuda()
                        max_index_c_list.append(max_index)
                # score_map_c_stack = torch.stack(score_map_c_list, dim=1)
                # max_index_c_stack = torch.stack(max_index_c_list, dim=0)
                # print("score_map_c_stack size:", score_map_c_stack.shape)
                # print("max_index_c_stack size:", max_index_c_stack.shape)
                # score_map_b_list.append(score_map_c_stack)
            # score_map_b_stack = torch.stack(score_map_b_list, dim=0)
    
    def backward_basic(self):
        self.loss_total = 0
        for batch_index in range(self.batch_size):
            loss_batch = 0
            point_count = 0
            for point_index in range(self.point_num):
                if self.correspondence_2D[batch_index][0][point_index][4] == -1:
                    continue
                
                # BCE loss
                feature_map1_divided = self.feature_map1[batch_index].unsqueeze(0)
                drr_POI = self.correspondence_2D[batch_index][0][point_index][0 : 4].clone()
                drr_POI[0] = torch.floor(drr_POI[0] / self.factor_H)  # No need to multiply factor
                drr_POI[1] = torch.floor(drr_POI[1] / self.factor_W)  # No need to multiply factor
                drr_POI[2] = torch.floor(drr_POI[2] / self.factor_H)  # No need to multiply factor
                drr_POI[3] = torch.floor(drr_POI[3] / self.factor_W)  # No need to multiply factor
                drr_POI = drr_POI.int()
                # print("self.factor_H", self.factor_H)
                # print("drr_POI:", drr_POI)
                # print("shape:", feature_map1_divided.shape)
                feature_kernel = torch.mul(self.FE_layer(feature_map1_divided, drr_POI), self.kernel_weight)
                # print(self.kernel_weight)
                # print("shape:", feature_kernel.shape)
                feature_map2_divided = self.feature_map2[batch_index].unsqueeze(0)
                score_map = F.conv2d(feature_map2_divided, feature_kernel, padding=self.patch_neighbor_size)
                score_map = self.batchnorm(score_map)
                score_map_unpadded = score_map.clone()
                # score_map = self.padding(score_map)
                # print("score_map size:", score_map.shape)
                # score_map = self.sigmoid(score_map)
                score_map_gt = self.generate_score_map_gt(drr_POI, self.f_size_H, self.f_size_W)

                # Eular distance loss
                score_map_squeezed = torch.squeeze(score_map)
                score_map_flattened = score_map_squeezed.view(-1)
                max_index_flattened = score_map_flattened.argmax(dim=0).float().requires_grad_(True)
                max_index = torch.tensor([max_index_flattened / self.f_size_H, max_index_flattened % self.f_size_W]).cuda().requires_grad_(True)
                # print("requires grad:", score_map_squeezed.requires_grad, max_index_flattened.requires_grad, max_index.requires_grad)
                # print("max_index:", max_index)
                eular_distance = torch.sqrt((max_index[0].float() - drr_POI[2].float()) ** 2 + (max_index[1].float() - drr_POI[3].float()) ** 2)
                print("eular distance:", eular_distance)
                # print("original eular distance:", torch.sqrt((drr_POI[0].float() - drr_POI[2].float()) ** 2 + (drr_POI[1].float() - drr_POI[3].float()) ** 2))
                
                # Sum
                loss_batch += eular_distance * 0.01
                loss_batch += self.criterion(score_map, score_map_gt)
                point_count += 1

                # -------------- Show ---------------- #
                # Find index of max value
                # score_map_squeezed = torch.squeeze(score_map)
                # score_map_flattened = score_map_squeezed.view(-1)
                # max_index_flattened = score_map_flattened.argmax(dim=0)
                # max_index = torch.tensor([max_index_flattened / self.f_size_H, max_index_flattened % self.f_size_W]).cuda()
                # max_index = max_index.int()
                # print("eular distance:", (max_index[0] - drr_POI[2]) ** 2 + (max_index[1] - drr_POI[3]) ** 2)
                # print("original eular distance:", (drr_POI[0] - drr_POI[2]) ** 2 + (drr_POI[1] - drr_POI[3]) ** 2)

                # score_map_squeezed_show = score_map_squeezed.cpu().data.numpy()
                # score_map_gt_show = torch.squeeze(score_map_gt)
                # score_map_gt_show = score_map_gt_show.cpu().data.numpy()
                # feature_map1_divided_show = torch.mean(feature_map1_divided, dim=1)
                # feature_map1_divided_show = torch.squeeze(feature_map1_divided_show)
                # feature_map1_divided_show = feature_map1_divided_show.cpu().data.numpy()
                # feature_map2_divided_show = torch.mean(feature_map2_divided, dim=1)
                # feature_map2_divided_show = torch.squeeze(feature_map2_divided_show)
                # feature_map2_divided_show = feature_map2_divided_show.cpu().data.numpy()
                # max_index_show = max_index.cpu().data.numpy()
                # drr_POI_show = drr_POI.cpu().data.numpy()
                # input1 = torch.squeeze(self.input_drr1[batch_index])
                # input1_show = input1.cpu().data.numpy()
                # input2 = torch.squeeze(self.input_drr2[batch_index])
                # input2_show = input2.cpu().data.numpy()
                # plt.subplot(231)
                # plt.imshow(input1_show, cmap='gray')
                # plt.scatter([drr_POI_show[1]], [drr_POI_show[0]], marker='+')
                # plt.title('DRR1')
                # plt.subplot(232)
                # plt.imshow(feature_map1_divided_show, cmap='gray')
                # plt.scatter([drr_POI_show[1]], [drr_POI_show[0]], marker='+')
                # plt.title('feature map1')
                # plt.subplot(233)
                # plt.imshow(score_map_gt_show, cmap='gray')
                # plt.title('point mask')
                # plt.subplot(234)
                # plt.imshow(input2_show, cmap='gray')
                # plt.scatter([drr_POI_show[3]], [drr_POI_show[2]], marker='x')
                # plt.title('DRR2')
                # plt.subplot(235)
                # plt.imshow(feature_map2_divided_show, cmap='gray')
                # plt.scatter([drr_POI_show[3]], [drr_POI_show[2]], marker='x')
                # plt.title('feature map2')
                # plt.subplot(236)
                # plt.imshow(score_map_squeezed_show, cmap='gray')
                # plt.scatter([drr_POI_show[1]], [drr_POI_show[0]], marker='+', cmap='orange')
                # plt.scatter([drr_POI_show[3]], [drr_POI_show[2]], marker='x', cmap='orange')
                # plt.scatter([max_index_show[1]], [max_index_show[0]], marker='o', cmap='green')
                # plt.title('score map')
                # plt.show()
            
            loss_batch = loss_batch / point_count
            self.loss_total += loss_batch
        
        self.loss_total = self.loss_total / self.batch_size
        print("loss:", self.loss_total)
        self.loss_total.backward()
        # print(self.UNet.up1.conv.double_conv[0].weight.grad)
    
    def optimize_parameters(self):
        # forward
        self()
        self.optimizer.zero_grad()
        self.backward_basic()
        nn.utils.clip_grad_value_(self.UNet.parameters(), 0.1)
        nn.utils.clip_grad_value_(self.kernel_weight, 0.1)
        self.optimizer.step()
