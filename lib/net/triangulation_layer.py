import torch
import torch.nn as nn
from kornia import SpatialSoftArgmax2d


class triangulation_layer(nn.Module):
    def __init__(self, device):
        super(triangulation_layer, self).__init__()
        self.device = device
        self.softArgmax = SpatialSoftArgmax2d(temperature=10000, normalized_coordinates=False) 

        # 成像参数
        self.distance = 1800
        self.center = 900  # 注意：c表示成像系统在Rt变换前的中心（初始化为焦距的一半）
        # self.K = troch.tensor([[self.distance, 0, 0],
        #                     [0, self.distance, 0],
        #                     [0, 0, 1]]).to(device=self.device, dtype=torch.float32)
        # self.h = troch.tensor([[0, 0, self.center]]).t().to(device=self.device, dtype=torch.float32)
        Tr_ap = torch.tensor([[1., 0., 0., 0.],
                                [0., 0., 1., -700.],
                                [0., -1., 0., 0.],
                                [0., 0., 0.,  1.]]).to(device=self.device, dtype=torch.float32)
        Tr_ap_inv = torch.inverse(Tr_ap)
        self.R_view_ap = Tr_ap_inv[0 : 3, 0 : 3]
        self.t_view_ap = Tr_ap_inv[:3, 3].t()
        Tr_lat = torch.tensor([[0., 0., 1., -700.],
                                [-1., 0., 0., 0.],
                                [0., -1., 0., 0.],
                                [0., 0., 0.,  1.]]).to(device=self.device, dtype=torch.float32)
        Tr_lat_inv = torch.inverse(Tr_lat)
        self.R_view_lat = Tr_lat_inv[0 : 3, 0 : 3]
        self.t_view_lat = Tr_lat_inv[:3, 3].t()
        self.center_volume = torch.tensor([127.5, 127.5, 127.5]).to(device=self.device, dtype=torch.float32)
        self.K_part = torch.tensor([[-self.distance, 0], [0, -self.distance]]).to(device=self.device, dtype=torch.float32)


    def forward(self, score_map_ap, score_map_lat):
        self.score_map_ap = score_map_ap
        self.score_map_lat = score_map_lat
        self.batch_size = score_map_ap.shape[0]
        self.point_num = score_map_ap.shape[1]
        self.s_size_H = score_map_ap.shape[2]
        self.s_size_W = score_map_ap.shape[3]

        fiducial_3D_pred_list = []
        for batch_index in range(self.batch_size):
            fiducial_3D_pred_per_batch_list = []
            for point_index in range(self.point_num):
                score_map_ap_devided = self.score_map_ap[batch_index][point_index]
                score_map_lat_devided = self.score_map_lat[batch_index][point_index]
                score_map_ap_devided = score_map_ap_devided.unsqueeze(0).unsqueeze(0)
                score_map_lat_devided = score_map_lat_devided.unsqueeze(0).unsqueeze(0)

                max_index_ap = self.softArgmax(score_map_ap_devided)
                max_index_ap = max_index_ap.view(-1)
                max_index_ap = torch.flip(max_index_ap, dims=[0])
                max_index_lat = self.softArgmax(score_map_lat_devided)
                max_index_lat = max_index_lat.view(-1)
                max_index_lat = torch.flip(max_index_lat, dims=[0])

                max_index_ap[0] = max_index_ap[0] - self.s_size_W / 2
                max_index_ap[1] = max_index_ap[1] - self.s_size_H / 2
                max_index_lat[0] = max_index_lat[0] - self.s_size_W / 2
                max_index_lat[1] = max_index_lat[1] - self.s_size_H / 2

                # max_index_ap = torch.tensor([86.40445959, -47.38309074]).to(device=self.device, dtype=torch.float32)
                # max_index_lat = torch.tensor([-15.55886736, -45.60357675]).to(device=self.device, dtype=torch.float32)

                D_x1 = torch.cat([self.K_part, max_index_ap.unsqueeze(0).t()], dim=1)
                D_x2 = torch.cat([self.K_part, max_index_lat.unsqueeze(0).t()], dim=1)

                A = torch.squeeze(torch.cat([torch.matmul(D_x1, self.R_view_ap), torch.matmul(D_x2, self.R_view_lat)], dim=0))
                b = torch.cat([-self.center * max_index_ap - torch.matmul(D_x1, self.t_view_ap), -self.center * max_index_lat - torch.matmul(D_x2, self.t_view_lat)], dim=0)

                X_3d_pred = torch.matmul(torch.pinverse(A), b)
                X_3d_pred = X_3d_pred + self.center_volume
                X_3d_pred = X_3d_pred.unsqueeze(0)

                # [77.5, 14.5, 42.5]

                fiducial_3D_pred_per_batch_list.append(X_3d_pred)

            fiducial_3D_pred_per_batch = torch.cat(fiducial_3D_pred_per_batch_list, dim=0)
            fiducial_3D_pred_per_batch = fiducial_3D_pred_per_batch.unsqueeze(0)
            fiducial_3D_pred_list.append(fiducial_3D_pred_per_batch)
        
        fiducial_3D_pred = torch.cat(fiducial_3D_pred_list, dim=0)

        return fiducial_3D_pred