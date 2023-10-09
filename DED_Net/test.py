import torch
import torchvision
from PIL import Image
import numpy as np
from glob import glob
import math
from nets.light_attnet_28layers import *


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


def test(image_path):

    data_lowlight = Image.open(image_path)

    data_lowlight = torchvision.transforms.ToTensor()(data_lowlight)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    LIS_Net = LISNet().cuda()
    NSS_Net = NSSNet().cuda()
    TCR_Net = TCRNet().cuda()

    checkpoint = torch.load("/home/liu/wzl/AEIE/AEIENet/checkpoint/LIS-Net_param.pth")
    LIS_Net.load_state_dict(checkpoint['A_net'])
    LIS_Net.eval()
    checkpoint = torch.load("/home/liu/wzl/AEIE/AEIENet/checkpoint/NSS-Net_param.pth")
    NSS_Net.load_state_dict(checkpoint['D_net'])
    checkpoint = torch.load("/home/liu/wzl/AEIE/AEIENet/checkpoint/TCR-Net_param.pth")
    TCR_Net.load_state_dict(checkpoint['C_net'])
    TCR_Net.eval()

    dark_image_HSV = rgb_to_hsv(data_lowlight)
    dark_image_H, dark_image_S, dark_image_V = torch.split(dark_image_HSV, 1, 1)

    attmap = LIS_Net(dark_image_V)

    LIS_V = dark_image_V / torch.clamp_min(2 * (1 - attmap), 1e-2)
    LIS_V = torch.clamp(LIS_V, 0., 1.)
    NSS_V = NSS_Net(LIS_V)

    TCR_image = TCR_Net(torch.cat([data_lowlight, dark_image_V, NSS_V], 1))

    TCR_H, TCR_S, TCR_V = torch.split(rgb_to_hsv(TCR_image), 1, 1)
    result_V = TCR_V * attmap + NSS_V * (1 - attmap)
    result_image = hsv_to_rgb(torch.cat([TCR_H, TCR_S, result_V], 1))

    result_path = image_path.replace('dark', 'a')
    result_path = result_path.replace('JPG', 'png')
    result_path = result_path.replace('jpg', 'png')

    torchvision.utils.save_image(result_image, result_path)


if __name__ == '__main__':
    with torch.no_grad():
        test_list = glob("/home/liu/wzl/Evaluation_Metrics/data/LOL/dark/*")
        for i in range(len(test_list)):
            print(test_list[i])
            test(test_list[i])
            