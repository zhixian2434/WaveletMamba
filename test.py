# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
import os
import argparse
from model import networks
import cv2
from model.wavelet import DWT, IWT
import time
from skimage import img_as_ubyte
import kornia

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def test(img):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda
    low_image = np.float32(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)) / 255.
    low_image = torch.from_numpy(low_image).permute(2, 0, 1)
    data_lowlight = low_image.cuda().unsqueeze(0)

    factor = 64
    b,c,h,w = data_lowlight.shape
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    dark = F.pad(data_lowlight, (0, padw, 0, padh), 'reflect')

    n, _, _, _ = dark.shape
    dark_dwt = dwt(dark)
    dark_LL, dark_high = dark_dwt[:n, ...], dark_dwt[n:, ...]

    result_LL, result_high = WMMNet(dark_LL, dark_high)

    pre_image = idwt(torch.cat((result_LL, result_high), dim=0))
    pre_image = pre_image[:, :, :h, :w]
    result = torch.clamp(pre_image, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    result_path = img.replace(img.split("/")[-2], 'result').replace(img.split("/")[-4], 'results')
    save_img(result_path, img_as_ubyte(result))

if __name__ == '__main__':
    with torch.no_grad():
        parser = argparse.ArgumentParser()
        parser.add_argument('--test_list', type=str, default="/home/liu/wzl/code/images/LOLv1/low/*")
        parser.add_argument('--cuda', type=str, default="0")
        parser.add_argument('--checkpoint', type=str, default="./checkpoint/BestEpoch.pth")

        config = parser.parse_args()

        dwt, idwt = DWT(), IWT()
        WMMNet = networks.WMMNet().cuda()
        WMMNet.load_state_dict(torch.load(config.checkpoint)["PreNet"])
        WMMNet.eval()

        test_list = glob(config.test_list)
        times = 0
        for image in test_list:
            start = time.time()
            print(image)
            test(image)
            end_time = (time.time() - start)
            times += end_time
        print("Time: %.3f " % (times / len(test_list)))
            