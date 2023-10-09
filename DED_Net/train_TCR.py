import torch
import torch.nn as nn
import torch.optim
import torchvision
import os
import argparse
import time
import dataloader_aeie
from nets.unet import Unet
import numpy as np
from math import cos, pi
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
#from utils.print_time import print_time
from tensorboardX import SummaryWriter
from nets.light_attnet_28layers import *
import cv2
import Myloss
from torch.autograd import Variable
import torch.nn.functional as F
import RGB_HSV_transformer
from glob import glob
from PIL import Image
from math import exp
import math
import Metric


def save_images(filepath, result_1, result_2=None):
    cat_image = result_1
    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')


def laplace(img):
    laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype='float32')
    laplace_kernel = laplace_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(laplace_kernel)).cuda()
    edge_detect = F.conv2d(Variable(img), weight, padding=1)
    edge_detect = edge_detect.detach()
    return edge_detect


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = 2.0 * math.pi * h  # we return 0/2pi output

    return torch.stack((h, s, v), dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out


def noise_syn(attmap):
    noise = torch.log10(1 + attmap)
    zero_copy = torch.zeros_like(noise)
    noise = torch.where(noise < 0, zero_copy, noise)
    return noise


def train(config):
    # Ours
    train_dark_path = glob("/home/liu/wzl/DataSet/LOL/train/low/*") + glob("/home/liu/wzl/DataSet/LOL/val/low/*") + glob("/home/liu/wzl/DataSet/LOL/train/high/*") + glob("/home/liu/wzl/DataSet/LOL/val/high/*")
    val_dark_path = glob("/home/liu/wzl/DataSet/LOL/val/low/*")
    train_en_path = glob("/home/liu/wzl/DataSet/LOL/train/en/*") + glob("/home/liu/wzl/DataSet/LOL/val/en/*") + glob("/home/liu/wzl/DataSet/LOL/train/en/*") + glob("/home/liu/wzl/DataSet/LOL/val/en/*")
    val_en_path = glob("/home/liu/wzl/DataSet/LOL/val/en/*")
    train_gth_path = glob("/home/liu/wzl/DataSet/LOL/train/high/*") + glob("/home/liu/wzl/DataSet/LOL/val/high/*") + glob("/home/liu/wzl/DataSet/LOL/train/high/*") + glob("/home/liu/wzl/DataSet/LOL/val/high/*")
    val_gth_path = glob("/home/liu/wzl/DataSet/LOL/val/high/*")

    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    CNet = TCRNet().cuda()

    total_params = sum(p.numel() for p in CNet.parameters() if p.requires_grad)
    print("Total_params: {}".format(total_params))

    train_datasets = dataloader_aeie.MyDataSet_Color(train_dark_path, train_en_path, train_gth_path)
    val_datasets = dataloader_aeie.MyDataSet_Color(val_dark_path, val_en_path, val_gth_path)
    train_data = DataLoader(train_datasets, batch_size=4, shuffle=True, num_workers=12, pin_memory=False)
    val_data = DataLoader(val_datasets, batch_size=1, shuffle=True, num_workers=12, pin_memory=False)

    print('dataset loaded!')
    print('%d images for training and %d images for evaluating.' % (len(train_data), len(val_data)))

    # loss function
    L_recon = Myloss.SSIML1Loss(channels=3)
    L_color = Myloss.color_loss()
    L_vgg = Myloss.VGG_LOSS().cuda()

    optimizer_D = torch.optim.Adam(CNet.parameters(), lr=config.start_lr, weight_decay=config.weight_decay)
    StepLR_D = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_D, step_size=50, gamma=0.5)

    # start training
    CNet.train()
    max_psnr = 0
    print("\nstart to train!")
    for epoch in range(config.num_epochs):
        index = 0
        # train
        for dark_image, en_image, gt_image in train_data:
            index += 1
            optimizer_D.zero_grad()
            dark_image = dark_image.cuda()
            en_image = en_image.cuda()
            gt_image = gt_image.cuda()

            dark_image_HSV = rgb_to_hsv(dark_image)
            dark_image_H, dark_image_S, dark_image_V = torch.split(dark_image_HSV, 1, 1)
            en_image_HSV= rgb_to_hsv(en_image)
            en_image_H, en_image_S, en_image_V = torch.split(en_image_HSV, 1, 1)

            result_image = CNet(torch.cat([dark_image, dark_image_V, en_image_V], 1))

            # train loss
            loss1 = L_recon(result_image, gt_image)
            loss2 = L_color(result_image, gt_image)
            loss3 = L_vgg(result_image, gt_image)

            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer_D.step()
        if epoch > 50:
            StepLR_D.step()

        # eval
        index = 0
        psnr_all = []
        ssim_all = []
        with torch.no_grad():
            CNet.eval()
            for dark_image, en_image, gt_image in val_data:
                index += 1
                dark_image = dark_image.cuda()
                en_image = en_image.cuda()
                gt_image = gt_image.cuda()

                dark_image_HSV = rgb_to_hsv(dark_image)
                dark_image_H, dark_image_S, dark_image_V = torch.split(dark_image_HSV, 1, 1)
                en_image_HSV= rgb_to_hsv(en_image)
                en_image_H, en_image_S, en_image_V = torch.split(en_image_HSV, 1, 1)

                result_image = CNet(torch.cat([dark_image, dark_image_V, en_image_V], 1))

                psnr = Metric.batch_PSNR(result_image, gt_image, 1.)
                ssim = Metric.SSIM(result_image, gt_image)
                psnr_all.append(psnr.item())
                ssim_all.append(ssim.item())

                print("epoch:", epoch, ",", "Loss at iteration", index, ":" , "PSNR:", psnr, "SSIM:", ssim)
                if ((index + 1) % 1) == 0:
                    torchvision.utils.save_image(gt_image, "/home/liu/wzl/AEIE/AEIENet/color_img/" + str(index) + "_gt.png")
                    torchvision.utils.save_image(result_image, "/home/liu/wzl/AEIE/AEIENet/color_img/" + str(index) + "_re.png")

            print("All_Epoch: ", epoch, "   PSNR: ", str(np.mean(np.array(psnr_all))), "   SSIM: ", str(np.mean(np.array(ssim_all))))
            if np.mean(np.array(ssim_all)) > max_psnr:
                max_psnr = np.mean(np.array(ssim_all))
                print("Best_All_Epoch: ", epoch, "   PSNR: ", str(np.mean(np.array(psnr_all))), "   SSIM: ", str(np.mean(np.array(ssim_all))))
            if ((epoch) % 1) == 0:
                state = {'C_net': CNet.state_dict()}
                torch.save(state, config.model_folder + "Epoch" + str(epoch) + '.pth')
            print("All_Epoch: ", epoch, "   SSIM: ", max_psnr)
        CNet.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--data_path', type=str, default="/home/liu/wzl/")
    parser.add_argument('--start_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_folder', type=str, default="model/ColorNet_model/new/")

    config = parser.parse_args()

    if not os.path.exists(config.model_folder):
        os.mkdir(config.model_folder)

    train(config)