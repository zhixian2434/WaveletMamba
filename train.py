# -*- coding: utf-8 -*-
from numpy import outer
import torch
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import os
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import networks
from pytorch_msssim import SSIM, MS_SSIM
import cv2
from model.wavelet import DWT, IWT
import torch.optim as optim

class MyDataSet(Dataset):
    def __init__(self, low, normal, mode):
        super(MyDataSet, self).__init__()
        self.low = low
        self.normal = normal
        self.mode = mode
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, item):
        name = self.low[item].split("/")[-1]

        low_image = np.float32(cv2.cvtColor(cv2.imread(self.low[item]), cv2.COLOR_BGR2RGB)) / 255.
        normal_image = np.float32(cv2.cvtColor(cv2.imread(self.normal[item]), cv2.COLOR_BGR2RGB)) / 255.

        if self.mode == "train":
            h, w, _ = low_image.shape
            x = np.random.randint(0, h - 256 + 1)
            y = np.random.randint(0, w - 256 + 1)

            low_image = low_image[x:x+256, y:y+256, :]
            normal_image = normal_image[x:x+256, y:y+256, :]
        
        low_image = torch.from_numpy(low_image).permute(2, 0, 1)
        normal_image = torch.from_numpy(normal_image).permute(2, 0, 1)

        return low_image, normal_image, name

    def __len__(self):
        return len(self.low)
    

class SSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(SSIML1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.4

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        total_loss = ssim_loss
        return l1_loss + self.alpha * total_loss


# I = R o L
# Ig = R o Lg
# Id = R o Ld + E
# Iadj = R o Ladj
# Iadj = (Id - E) o Ladj / Ld
# Iadj = Id * L_ - E * L_     where L_ 表示Ladj/Ld, E_表示噪声、伪影、过曝/欠曝光、颜色偏移等退化。


def train():

    train_dark_path = glob("/home/liu/wzl/Dataset/syn/*") + glob("/home/liu/wzl/Dataset/Data/LOL/train/low/*") + glob("/home/liu/wzl/Dataset/Data/LOL-v2/train/low/*") + glob("/home/liu/wzl/Dataset/Data/LOLv2-SYS/Train/low/*")
    val_dark_path = glob("/home/liu/wzl/Dataset/Data/LOL/val/low/*")
    train_gth_path = glob("/home/liu/wzl/Dataset/high/*") + glob("/home/liu/wzl/Dataset/Data/LOL/train/high/*") + glob("/home/liu/wzl/Dataset/Data/LOL-v2/train/high/*") + glob("/home/liu/wzl/Dataset/Data/LOLv2-SYS/Train/high/*")
    val_gth_path = glob("/home/liu/wzl/Dataset/Data/LOL/val/high/*")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    net_Pre = networks.WMMNet().cuda()

    train_datasets = MyDataSet(train_dark_path, train_gth_path, mode="train")
    val_datasets = MyDataSet(val_dark_path, val_gth_path, mode="val")
    train_data = DataLoader(train_datasets, batch_size=12, shuffle=True, num_workers=8, drop_last=True)
    val_data = DataLoader(val_datasets, batch_size=1, shuffle=True, num_workers=8, drop_last=True)

    print('dataset loaded!')
    print('%d images for training and %d images for evaluating.' % (len(train_data), len(val_data)))
    
    optimizer_Pre = optim.Adam(net_Pre.parameters(), lr=2e-4, betas=[0.9, 0.999], eps=0.00000001)
    scheduler = optim.lr_scheduler.StepLR(optimizer_Pre, step_size=500,
                                          gamma=0.8, last_epoch=-1)

    net_Pre.train()

    L_sl1 = SSIML1Loss(channels=3)
    L_l2 = nn.L1Loss()
    dwt, idwt = DWT(), IWT()

    min_loss = 99999
    for epoch in range(2000):
        index = 0
        # train
        for dark_image, target_image, name in train_data:
            index += 1
            optimizer_Pre.zero_grad()

            dark_image = dark_image.cuda()
            target_image = target_image.cuda()

            n, _, _, _ = dark_image.shape
            dark_dwt = dwt(dark_image)
            dark_LL, dark_high = dark_dwt[:n, ...], dark_dwt[n:, ...]
            target_dwt = dwt(target_image)
            target_LL, target_high = target_dwt[:n, ...], target_dwt[n:, ...]

            resultLL, resulthigh = net_Pre(dark_LL, dark_high)
            result = idwt(torch.cat((resultLL, resulthigh), dim=0))

            loss1 = L_sl1(result, target_image) + L_l2(resultLL, target_LL) + L_l2(resulthigh, target_high)
            loss = loss1

            loss.backward()
            optimizer_Pre.step()
        
        scheduler.step()
        # eval
        index = 0
        factor = 64
        val_loss = 0
        with torch.no_grad():
            net_Pre.eval()
            for dark_image, target_image, name in val_data:
                index += 1

                dark_image = dark_image.cuda()
                target_image = target_image.cuda()

                b,c,h,w = dark_image.shape
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                dark = F.pad(dark_image, (0, padw, 0, padh), 'reflect')
                target = F.pad(target_image, (0, padw, 0, padh), 'reflect')

                n, _, _, _ = dark.shape
                dark_dwt = dwt(dark)
                dark_LL, dark_high = dark_dwt[:n, ...], dark_dwt[n:, ...]
                target_dwt = dwt(target)
                target_LL, target_high = target_dwt[:n, ...], target_dwt[n:, ...]

                resultLL, resulthigh = net_Pre(dark_LL, dark_high)
                result = idwt(torch.cat((resultLL, resulthigh), dim=0))
                result = result[:, :, :h, :w]
                
                loss1 = L_sl1(result, target_image) + L_l2(resultLL, target_LL) + L_l2(resulthigh, target_high)
                loss = loss1

                val_loss += loss

            val_loss = val_loss / len(val_data)
            print(val_loss)
            state = {'PreNet': net_Pre.state_dict()}
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(state,  "/home/liu/wzl/code/checkpoint/BestEpochx.pth", _use_new_zipfile_serialization=False)
                print('saving the best epoch %d model with the loss %.5f' % (epoch + 1, min_loss))
        net_Pre.train()


if __name__ == "__main__":
    train()