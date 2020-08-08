import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from lib.net.POINT_model import PNet
from lib.net.POINT2_model import P2Net

from lib.dataset.alignDataSet import AlignDataSet
from torch.utils.data import DataLoader, random_split

dir_data = "/home/liulilu/POINT2-data/data_multiview_cq500_train"
dir_checkpoint = "./checkpoints/"

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):
    dataset = AlignDataSet(dir_data)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    for epoch in range(epochs):
        print(f"Start {epoch}th epoch!")
        net.train()
        
        epoch_loss = 0
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            print("Start {}th epoch, {}th batch!".format(epoch, batch_count))

            input_drr_ap = batch[0].to(device=device, dtype=torch.float32)
            input_xray_ap = batch[1].to(device=device, dtype=torch.float32)
            correspondence_2D_ap = batch[2].to(device=device, dtype=torch.float32)
            input_drr_lat = batch[3].to(device=device, dtype=torch.float32)
            input_xray_lat = batch[4].to(device=device, dtype=torch.float32)
            correspondence_2D_lat = batch[5].to(device=device, dtype=torch.float32)
            fiducial_3D = batch[6].to(device=device, dtype=torch.float32)
            # print("shape:", input_drr1.shape, input_drr2.shape, correspondence_2D.shape)

            net.set_input(input_drr_ap, input_xray_ap, correspondence_2D_ap, 
                            input_drr_lat, input_xray_lat, correspondence_2D_lat, fiducial_3D)

            net.optimize_parameters()

        torch.save(net.state_dict(), './checkpoints_cq500_with_triangle/checkpoint_{}.pth'.format(epoch))



if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = P2Net(device=device, n_channels=1, bilinear=True)

    net.load_state_dict((torch.load("/home/liulilu/POINT2-pytorch/checkpoints_cq500_0807/checkpoint_99.pth", map_location=device)))

    net.to(device=device)

    try:
        train_net(net=net,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    lr=args.lr,
                    device=device,
                    img_scale=args.scale,
                    val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
