import torch
import numpy as np
import cv2
from glob import glob
import torchvision
import kornia
import torch.nn.functional as F
from skimage import img_as_ubyte
from natsort import natsorted

def guide_filter(I, p, r, eps):
    """
    引导滤波实现
    :param I: 引导图像，Tensor of shape (N, C, H, W)
    :param p: 输入图像，Tensor of shape (N, C, H, W)
    :param r: 局部窗口半径
    :param eps: 正则化参数
    :return: 输出图像，Tensor of shape (N, C, H, W)
    """
    # 计算I,p的均值
    mean_I = F.avg_pool2d(I, kernel_size=2*r+1, stride=1, padding=r)
    mean_p = F.avg_pool2d(p, kernel_size=2*r+1, stride=1, padding=r)
    
    # 计算I*p的均值
    mean_Ip = F.avg_pool2d(I*p, kernel_size=2*r+1, stride=1, padding=r)
    # 计算I的方差和I,p的协方差
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = F.avg_pool2d(I*I, kernel_size=2*r+1, stride=1, padding=r) - mean_I * mean_I
    
    # 计算系数a和b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # 计算a,b的均值
    mean_a = F.avg_pool2d(a, kernel_size=2*r+1, stride=1, padding=r)
    mean_b = F.avg_pool2d(b, kernel_size=2*r+1, stride=1, padding=r)
    
    # 构造输出图像
    q = mean_a * I + mean_b
    return q

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def cal_dark(i1, i2, i3):
    r = 64
    eps = 1e-4
    name1 = i1.split("/")[-1]
    name2 = i2.split("/")[-1]
    z = int(name1.split(".")[0].split("_")[-1])
    alpha = 0.5#(1 + math.cos(math.pi * (z / 6))) / 2
    print(name1, name2, alpha)

    img1 = torch.from_numpy(np.float32(cv2.cvtColor(cv2.imread(i1), cv2.COLOR_BGR2RGB)) / 255.).permute(2, 0, 1).cuda().unsqueeze(0)
    H1, S1, V1 = torch.split(kornia.color.rgb_to_hsv(img1), 1, 1)
    img2 = torch.from_numpy(np.float32(cv2.cvtColor(cv2.imread(i2), cv2.COLOR_BGR2RGB)) / 255.).permute(2, 0, 1).cuda().unsqueeze(0)
    img2 = torchvision.transforms.Resize([400, 600])(img2)
    H2, S2, V2 = torch.split(kornia.color.rgb_to_hsv(img2), 1, 1)
    Q = guide_filter(V1, V2, r, eps)

    img3 = torch.from_numpy(np.float32(cv2.cvtColor(cv2.imread(i3), cv2.COLOR_BGR2RGB)) / 255.).permute(2, 0, 1).cuda().unsqueeze(0)
    H3, S3, V3 = torch.split(kornia.color.rgb_to_hsv(img3), 1, 1)

    result = kornia.color.hsv_to_rgb(torch.cat([H3, S3, alpha*V3 + (1 - alpha)*Q], 1))

    result_path = "./Dataset/syn/" + name1
    save_img(result_path, img_as_ubyte(result))


if __name__ == "__main__":
    high_list = natsorted(glob("./LOL/train/high/*"))
    dark_list = natsorted(glob("./LOL/train/low/*"))
    low_list = natsorted(glob("./Non-uniform/resize_input/*"))
    
    cal_dark(high_list, low_list, dark_list)


