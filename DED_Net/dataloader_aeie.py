import os
import math
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision
import numpy as np
from PIL import Image
import random
import cv2
import torch


def data_aug(img1, img2):
    a = random.random()
    b = math.floor(random.random() * 4)
    if a >= 0.5:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if b == 1:
        img1 = img1.transpose(Image.ROTATE_90)
        img2 = img2.transpose(Image.ROTATE_90)
    elif b == 2:
        img1 = img1.transpose(Image.ROTATE_180)
        img2 = img2.transpose(Image.ROTATE_180)
    elif b == 3:
        img1 = img1.transpose(Image.ROTATE_270)
        img2 = img2.transpose(Image.ROTATE_270)
    return img1, img2


class DataSet_loader(data.Dataset):

    def __init__(self, transform1, is_gth_train, path=None, flag='train'):
        self.flag = flag
        self.transform1 = transform1
        self.dark_path, self.gt_path = path
        self.dark_data_list = os.listdir(self.dark_path)
        self.dark_data_list.sort(key=lambda x: int(x[:-4]))
        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))
        if is_gth_train:
            self.dark_data_list = self.dark_data_list + self.gt_data_list
        self.length = len(self.dark_data_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dark_name = self.dark_data_list[idx][:-4]
        tail = self.dark_data_list[idx][-4:]
        gth_name = dark_name
        gt_image = Image.open(self.gt_path + gth_name + tail)
        dark_image = Image.open(self.dark_path + dark_name + tail)
        # 数据增强
        """if self.flag == 'train':
            dark_image, gt_image = data_aug(dark_image, gt_image)"""

        """dark_image = np.uint8(dark_image)
        orgimg_HSV = cv2.cvtColor(dark_image, cv2.COLOR_BGR2HSV)
        org_H, org_S, org_V = cv2.split(orgimg_HSV)
        clahev = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        new_V = clahev.apply(org_V)
        new_V[new_V < 0] = 0.0
        new_V[new_V > 255.0] = 255.0
        new_V = new_V.astype(np.uint8)
        rgb_img = cv2.merge([org_H, org_S, new_V])
        dark_image = cv2.cvtColor(rgb_img, cv2.COLOR_HSV2BGR)

        gt_image = np.uint8(gt_image)
        orgimg_HSV = cv2.cvtColor(gt_image, cv2.COLOR_BGR2HSV)
        org_H, org_S, org_V = cv2.split(orgimg_HSV)
        clahev = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        new_V = clahev.apply(org_V)
        new_V[new_V < 0] = 0.0
        new_V[new_V > 255.0] = 255.0
        new_V = new_V.astype(np.uint8)
        rgb_img = cv2.merge([org_H, org_S, new_V])
        gt_image = cv2.cvtColor(rgb_img, cv2.COLOR_HSV2BGR)"""

        dark_image = np.asarray(dark_image)
        gt_image = np.asarray(gt_image)

        if self.transform1:
            dark_image = self.transform1(dark_image)
            gt_image = self.transform1(gt_image)

        return dark_name, dark_image, gt_image
    

class MyDataSet_Color(Dataset):
    def __init__(self, low, en, normal):
        super(MyDataSet_Color, self).__init__()
        self.low = low
        self.en = en
        self.normal = normal
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, item):
        low_image = Image.open(self.low[item])
        en_image = Image.open(self.en[item])
        normal_image = Image.open(self.normal[item])
        
        low_image = self.transform(low_image)
        en_image = self.transform(en_image)
        normal_image = self.transform(normal_image)

        return low_image, en_image, normal_image

    def __len__(self):
        return len(self.low)


class MyDataSet(Dataset):
    def __init__(self, low, normal):
        super(MyDataSet, self).__init__()
        self.low = low
        self.normal = normal
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, item):
        low_image = Image.open(self.low[item])
        normal_image = Image.open(self.normal[item])
        
        low_image = self.transform(low_image)
        normal_image = self.transform(normal_image)

        return low_image, normal_image

    def __len__(self):
        return len(self.low)
