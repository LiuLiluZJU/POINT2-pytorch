import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from kornia import SpatialSoftArgmax2d
from .unet_model import UNet
from .FE_layer import FE_layer
import matplotlib.pyplot as plt
import math

from graphviz import Digraph
from torch.autograd import Variable, Function


def gaussian2d(mu, sigma, shape):
    (r, c), (sr, sc), (H, W) = mu, sigma, shape
    pi = torch.tensor(math.pi)
    rr = torch.arange(r - 3 * sr, r + 3 * sr + 1).float()
    cc = torch.arange(c - 3 * sc, c + 3 * sc + 1).float()
    rr = rr[(rr >= 0) & (rr < H)]
    cc = cc[(cc >= 0) & (cc < W)]
    gr = torch.exp(-0.5 * ((rr - r) / sr)**2) / (torch.sqrt(2 * pi) * sr)
    gc = torch.exp(-0.5 * ((cc - c) / sc)**2) / (torch.sqrt(2 * pi) * sc)
    g = torch.ger(gr, gc).view(-1)
    rr, cc = torch.meshgrid(rr.long(), cc.long())
    rr = rr.contiguous().view(-1)
    cc = cc.contiguous().view(-1)
    return rr, cc, g


class PNet(nn.Module):
    def __init__(self, device, n_channels=1, n_classes=64, bilinear=True, patch_neighbor_size=10):
        super(PNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_neighbor_size = patch_neighbor_size
        self.UNet = UNet(self.n_channels, self.n_classes, bilinear=True)
        self.FE_layer = FE_layer(patch_neighbor_size)
        self.padding = nn.ReplicationPad2d(patch_neighbor_size)
        # self.loss = nn.BCELoss()
        self.batchnorm = nn.BatchNorm2d(1)
        self.kernel_weight = nn.Parameter(torch.ones((1, self.n_classes, 2 * patch_neighbor_size + 1, 2 * patch_neighbor_size + 1)))
        self.softArgmax = SpatialSoftArgmax2d(temperature=10000, normalized_coordinates=False) 
        self.optimizer = torch.optim.Adam([
            {'params': self.UNet.parameters()},
            {'params': self.kernel_weight}
            ], lr=0.001, weight_decay=1e-8)
        self.device = device
    
    def set_input(self, input1, input2, correspondence_2D):
        self.input_drr1 = input1
        self.input_drr2 = input2
        self.correspondence_2D = correspondence_2D
        self.batch_size = self.correspondence_2D.shape[0]
        self.point_num = self.correspondence_2D.shape[2]
        print(self.correspondence_2D.shape)

    def generate_score_map_gt(self, single_POI, size_H, size_W):

        # score_map_gt = torch.zeros((size_H, size_W)).cuda()
        # # score_map_gt = score_map_gt * (-10)
        # up_bound = single_POI[2] + self.patch_neighbor_size
        # low_bound = single_POI[2] - self.patch_neighbor_size
        # right_bound = single_POI[3] + self.patch_neighbor_size
        # left_bound = single_POI[3] - self.patch_neighbor_size

        # if up_bound >= size_H:
        #     up_bound = size_H - 1
        # if low_bound < 0:
        #     low_bound = 0
        # if right_bound >= size_W:
        #     right_bound = size_W - 1
        # if left_bound < 0:
        #     left_bound = 0
        
        # score_map_gt[low_bound : up_bound + 1, left_bound : right_bound + 1] = 1
        # score_map_gt = score_map_gt.unsqueeze(0).unsqueeze(0)

        score_map_gt = torch.zeros((size_H, size_W))
        rr, cc, g = gaussian2d([single_POI[2], single_POI[3]], [self.patch_neighbor_size, self.patch_neighbor_size], shape=score_map_gt.shape)
        score_map_gt[rr, cc] = torch.max(score_map_gt[rr, cc], g / g.max())
        score_map_gt = score_map_gt.unsqueeze(0).unsqueeze(0)
        score_map_gt = score_map_gt.clone()
        score_map_gt = score_map_gt.to(device=self.device, dtype=torch.float32)
        # print("max, min:", torch.max(score_map_gt), torch.min(score_map_gt))

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
        
        score_map_total_list = []
        score_map_gt_total_list = []
        for batch_index in range(self.batch_size):
            score_map_per_batch_list = []
            score_map_gt_per_batch_list = []
            for point_index in range(self.point_num):

                drr_POI = self.correspondence_2D[batch_index][0][point_index][0 : 4].clone()
                drr_POI[0] = torch.floor(drr_POI[0] / self.factor_H)  # No need to multiply factor
                drr_POI[1] = torch.floor(drr_POI[1] / self.factor_W)  # No need to multiply factor
                drr_POI[2] = torch.floor(drr_POI[2] / self.factor_H)  # No need to multiply factor
                drr_POI[3] = torch.floor(drr_POI[3] / self.factor_W)  # No need to multiply factor
                drr_POI = drr_POI.int()
                
                # extract feature
                feature_map1_divided = self.feature_map1[batch_index].unsqueeze(0)
                feature_kernel = torch.mul(self.FE_layer(feature_map1_divided, drr_POI), self.kernel_weight)
                
                # get score map
                feature_map2_divided = self.feature_map2[batch_index].unsqueeze(0)
                score_map = F.conv2d(feature_map2_divided, feature_kernel)
                upsample = nn.Upsample(scale_factor=(self.f_size_H / score_map.shape[2], self.f_size_W / score_map.shape[3]), mode='bilinear', align_corners=True)
                score_map = upsample(score_map)
                score_map = self.batchnorm(score_map)
                score_map_per_batch_list.append(score_map)

                # get ground truth score map
                score_map_gt = self.generate_score_map_gt(drr_POI, self.f_size_H, self.f_size_W)
                score_map_gt_per_batch_list.append(score_map_gt)

                # -------------- Show ---------------- #
                show_flag = 0
                if show_flag == 1:
                    # Soft Argmax
                    max_index = self.softArgmax(score_map)
                    max_index_flattened = max_index.view(-1)
                    max_index_flattened = torch.flip(max_index_flattened, dims=[0])

                    score_map_squeezed = torch.squeeze(score_map)
                    score_map_flattened = score_map_squeezed.view(-1)
                    print("original eular distance:", (drr_POI[0] - drr_POI[2]) ** 2 + (drr_POI[1] - drr_POI[3]) ** 2)

                    score_map_squeezed_show = score_map_squeezed.cpu().data.numpy()
                    feature_map1_divided_show = torch.mean(feature_map1_divided, dim=1)
                    feature_map1_divided_show = torch.squeeze(feature_map1_divided_show)
                    feature_map1_divided_show = feature_map1_divided_show.cpu().data.numpy()
                    feature_map2_divided_show = torch.mean(feature_map2_divided, dim=1)
                    feature_map2_divided_show = torch.squeeze(feature_map2_divided_show)
                    feature_map2_divided_show = feature_map2_divided_show.cpu().data.numpy()
                    max_index_show = max_index_flattened.cpu().data.numpy()
                    drr_POI_show = drr_POI.cpu().data.numpy()
                    input1 = torch.squeeze(self.input_drr1[batch_index])
                    input1_show = input1.cpu().data.numpy()
                    input2 = torch.squeeze(self.input_drr2[batch_index])
                    input2_show = input2.cpu().data.numpy()

                    fig = plt.figure()
                    plt.subplot(221)
                    plt.imshow(input1_show, cmap='gray')
                    plt.scatter([drr_POI_show[1]], [drr_POI_show[0]], marker='+')
                    plt.title('DRR1')
                    plt.subplot(223)
                    plt.imshow(input2_show, cmap='gray')
                    plt.scatter([drr_POI_show[3]], [drr_POI_show[2]], marker='x')
                    plt.title('DRR2')
                    plt.subplot(222)
                    plt.imshow(feature_map2_divided_show, cmap='gray')
                    plt.scatter([drr_POI_show[3]], [drr_POI_show[2]], marker='x')
                    plt.title('feature map')
                    plt.subplot(224)
                    plt.imshow(score_map_squeezed_show, cmap='gray')
                    plt.scatter([drr_POI_show[1]], [drr_POI_show[0]], marker='+', cmap='orange')
                    plt.scatter([drr_POI_show[3]], [drr_POI_show[2]], marker='x', cmap='orange')
                    plt.scatter([max_index_show[1]], [max_index_show[0]], marker='o', cmap='green')
                    plt.title('score map')
                    fig.tight_layout()
                    plt.show()
            
            score_map_per_batch = torch.cat(score_map_per_batch_list, dim=1)
            score_map_total_list.append(score_map_per_batch)
            score_map_gt_per_batch = torch.cat(score_map_gt_per_batch_list, dim=1)
            score_map_gt_total_list.append(score_map_gt_per_batch)
        
        self.score_map_total = torch.cat(score_map_total_list, dim=0)
        self.score_map_gt_total = torch.cat(score_map_gt_total_list, dim=0)

        # # Soft Argmax
        # self.soft_max_index = self.softArgmax(self.score_map_total)
        # print(self.soft_max_index.shape)

        return self.score_map_total, self.score_map_gt_total

    
    def backward_basic(self):

        # Sum
        self.loss_total = F.binary_cross_entropy_with_logits(self.score_map_total, self.score_map_gt_total, reduction='mean')

        # g = make_dot(self.loss_total)
        # g.view()
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
