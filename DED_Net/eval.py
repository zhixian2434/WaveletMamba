import glob
import torch
import torchvision
import numpy as np
from PIL import Image
import math
import os
from glob import glob
import Metric
import lpips


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


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def PSNR_SSIM():
    en_img = glob("/home/liu/wzl/Evaluation_Metrics/data/LOL/a/*")
    psnrs = []
    ssims = []
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    lpip = []
    for en in en_img:
        filepath, tempname = os.path.split(en)
        imgname, extension = os.path.splitext(tempname)
        imgname = imgname.replace("_Bread", "")
        gt = "/home/liu/wzl/Evaluation_Metrics/data/LOL/gt/" + imgname + ".png"
        en = torchvision.transforms.ToTensor()(Image.open(en)).unsqueeze(0).cuda()
        gt = torchvision.transforms.ToTensor()(Image.open(gt)).unsqueeze(0).cuda()
        
        psnr = Metric.batch_PSNR(gt, en, 1.)
        ssim = Metric.SSIM(en, gt).item()
        lp = loss_fn_alex(en, gt).item()
        psnrs.append(psnr)
        ssims.append(ssim)
        lpip.append(lp)
        print(tempname, "PSNR->", psnr, "   SSIM->", ssim, "    LPIPS->", lp)
    print("最终PSNR: ", np.mean(np.array(psnrs)), "   SSIM: ", np.mean(np.array(ssims)), "   LPIPS: ", np.mean(np.array(lpip)))

if __name__ == '__main__':
    PSNR_SSIM()