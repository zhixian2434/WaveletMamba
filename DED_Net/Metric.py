import math
from math import exp
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import random


def noise_syn(attmap, strength):
    noise = strength * torch.log(1 + attmap)
    zero_copy = torch.zeros_like(noise)
    noise = torch.where(noise < 0, zero_copy, noise)
    return noise


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


def SSIM(x_image, y_image, max_value=1.0, win_size=3, use_sample_covariance=True):
    x_image = x_image.permute(0, 2, 3, 1)
    y_image = y_image.permute(0, 2, 3, 1)
    x = x_image.data.cpu().numpy().astype(np.float32)
    y = y_image.data.cpu().numpy().astype(np.float32)
    ssim=0
    for i in range(x.shape[0]):
        ssim += structural_similarity(x[i,:,:,:],y[i,:,:,:], win_size=win_size, data_range=max_value, multichannel=True)
    return (ssim/x.shape[0])

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


class add_noise(nn.Module):
    def __init__(self):
        super().__init__()
    def Gauss_noise(self, img, attention):
        noise = torch.normal(mean=0., std=attention)
        noise_img = img + noise
        noise_img = torch.clamp(noise_img, 0.0, 1.0).cuda()
        return noise_img
    def Poss_noise(self, img, gt_img):
        strength = np.random.uniform(2.5, 4)
        t = 10 * torch.ones_like(gt_img)
        x = torch.poisson(gt_img * torch.pow(t, strength)) / torch.pow(t, strength)
        noise = gt_img - x
        noise_img = img + noise
        noise_img = torch.clamp(noise_img, 0.0, 1.0)
        return noise_img
    def Speck_noise(self, img, attention):
        noise = torch.normal(mean=0., std=attention)
        noise_img = img + noise * img
        noise_img = torch.clamp(noise_img, 0.0, 1.0).cuda()
        return noise_img
    def Salt_noise(self, img):
        noise = torch.rand_like(img)
        x = torch.ones_like(noise)
        y = torch.zeros_like(noise)
        noise = torch.where(noise > 0.998, x, y)
        noise_img = torch.clamp(img + noise, 0.0, 1.0)
        return noise_img
    def forward(self, ori_img, gt_img, attention):
        # 高斯噪声一定会添加1次，其他的噪声可能出现1次，也可能不出现,顺序随机
        noise_index = [1, 2, 3]
        n = np.random.randint(0, 4)
        index = random.sample(noise_index, n)
        index.append(0)
        index = random.sample(index, n + 1)
        noise_img = ori_img
        # 添加噪声
        for idx in index:
            if idx == 0:
                noise_img = self.Gauss_noise(noise_img, attention)
            elif idx == 1:
                noise_img = self.Poss_noise(noise_img, gt_img)
            elif idx == 2:
                noise_img = self.Speck_noise(noise_img, attention)
            else:
                noise_img = self.Salt_noise(noise_img)
        
        return noise_img
    

class N_Add(nn.Module):
    def __init__(self):
        super().__init__()
    def Gauss_noise(self, img):
        attention = torch.ones_like(img) * np.random.randint(1, 21) / 255.0
        noise = torch.normal(mean=0., std=attention)
        return noise
    def Poss_noise(self, img, gt_img):
        strength = np.random.uniform(30, 41) / 10.0
        t = 10 * torch.ones_like(gt_img)
        x = torch.poisson(gt_img * torch.pow(t, strength)) / torch.pow(t, strength)
        noise = gt_img - x
        return noise
    def Speck_noise(self, img):
        attention = torch.ones_like(img) * np.random.randint(1, 21) / 255.0
        noise = torch.normal(mean=0., std=attention)
        noise = noise * img
        return noise
    def Salt_noise(self, img):
        noise = torch.rand_like(img)
        x = torch.ones_like(noise)
        y = torch.zeros_like(noise)
        noise = torch.where(noise > 0.998, x, y)
        return noise
    def Resize(self, img1, img2, img3, img4):
        attention = np.random.randint(5, 21) / 10.0
        randmode = np.random.randint(0, 2)
        if randmode == 0:
            img1 = F.interpolate(img1, scale_factor=attention, mode='bicubic', align_corners=True)
            img2 = F.interpolate(img2, scale_factor=attention, mode='bicubic', align_corners=True)
            img3 = F.interpolate(img3, scale_factor=attention, mode='bicubic', align_corners=True)
            img4 = F.interpolate(img4, scale_factor=attention, mode='bicubic', align_corners=True)
        else:
            img1 = F.interpolate(img1, scale_factor=attention, mode='bilinear', align_corners=True)
            img2 = F.interpolate(img2, scale_factor=attention, mode='bilinear', align_corners=True)
            img3 = F.interpolate(img3, scale_factor=attention, mode='bilinear', align_corners=True)
            img4 = F.interpolate(img4, scale_factor=attention, mode='bilinear', align_corners=True)
        return img1, img2, img3, img4

    def forward(self, ori_img, gt_img, low_img):
        # 高斯噪声一定会添加1次，其他的噪声可能出现1次，也可能不出现,顺序随机
        noise_index = [1, 1, 2]
        n = np.random.randint(0, 4)
        index = random.sample(noise_index, n)
        index.append(0)
        index.append(0)
        index = random.sample(index, n + 1)
        noise_img = ori_img
        # 添加噪声
        for idx in index:
            if idx == 0:
                noise_img += self.Gauss_noise(ori_img)
                noise_img = torch.clamp(noise_img, 0.0, 1.0)
            elif idx == 1:
                noise_img += self.Poss_noise(ori_img, gt_img)
                noise_img = torch.clamp(noise_img, 0.0, 1.0)
            else:
                noise_img += self.Speck_noise(ori_img)
                noise_img = torch.clamp(noise_img, 0.0, 1.0)
        
        return noise_img, gt_img, low_img


