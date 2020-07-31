import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from kornia import SpatialSoftArgmax2d
from .unet_model import UNet
from .FE_layer import FE_layer
from .POINT_model import PNet
from .triangulation_layer import triangulation_layer
import matplotlib.pyplot as plt
import math

from graphviz import Digraph
from torch.autograd import Variable, Function


class P2Net(nn.Module):
    def __init__(self, device, n_channels=1, n_classes=64, bilinear=True, patch_neighbor_size=10):
        super(P2Net, self).__init__()
        self.device = device
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_neighbor_size = patch_neighbor_size
        self.PNet_ap = PNet(self.device, self.n_channels, self.n_classes, self.bilinear, self.patch_neighbor_size)
        self.PNet_lat = PNet(self.device, self.n_channels, self.n_classes, self.bilinear, self.patch_neighbor_size)
        self.triangulation_layer = triangulation_layer(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.PNet_ap.parameters()},
            {'params': self.PNet_lat.parameters()}
            ], lr=0.001, weight_decay=1e-8)
        
    
    def set_input(self, input_drr_ap, input_xray_ap, correspondence_2D_ap, 
                        input_drr_lat, input_xray_lat, correspondence_2D_lat, fiducial_3D):
        self.input_drr_ap = input_drr_ap
        self.input_xray_ap = input_xray_ap
        self.correspondence_2D_ap = correspondence_2D_ap
        self.input_drr_lat = input_drr_lat
        self.input_xray_lat = input_xray_lat
        self.correspondence_2D_lat = correspondence_2D_lat
        self.fiducial_3D = fiducial_3D
        self.batch_size = self.correspondence_2D_ap.shape[0]
        self.point_num = self.correspondence_2D_ap.shape[2]
        self.PNet_ap.set_input(self.input_drr_ap, self.input_xray_ap, self.correspondence_2D_ap)
        self.PNet_lat.set_input(self.input_drr_lat, self.input_xray_lat, self.correspondence_2D_lat)

        
    def forward(self):
        
        self.score_map_ap, self.score_map_gt_ap = self.PNet_ap()
        self.score_map_lat, self.score_map_gt_lat = self.PNet_lat()

        self.fiducial_3D_pred = self.triangulation_layer(self.score_map_ap, self.score_map_lat)

    
    def backward_basic(self):

        self.loss_bce_ap = F.binary_cross_entropy_with_logits(self.score_map_ap, self.score_map_gt_ap, reduction='mean')
        self.loss_bce_lat = F.binary_cross_entropy_with_logits(self.score_map_lat, self.score_map_gt_lat, reduction='mean')
        self.loss_eular = torch.norm(self.fiducial_3D_pred - self.fiducial_3D)
        print("self.loss_eular", self.loss_eular)
        
        self.loss_total = self.loss_eular
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
        nn.utils.clip_grad_value_(self.PNet_ap.parameters(), 0.1)
        nn.utils.clip_grad_value_(self.PNet_lat.parameters(), 0.1)
        self.optimizer.step()
