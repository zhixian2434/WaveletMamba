import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import Metric


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


class D_Net(nn.Module):

    def __init__(self):
        super(D_Net, self).__init__()

        self.relu = nn.LeakyReLU(inplace=True)
        self.sig = nn.Sigmoid()

        # Number of channels in the middle hidden layer
        number_f = 16

        # attmap estimate net
        self.e_conv1 = nn.Conv2d(1, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)                  
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv8 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv9 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv10 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv11 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv12 = nn.Conv2d(number_f, 1, 3, 1, 1, bias=True)   
        

    def forward(self, dark_image_V):
        
        dark_V = nn.ReflectionPad2d(4)(dark_image_V)

        # attmap estimate net
        x1 = self.relu(self.e_conv1(dark_V))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x4, x3], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x5, x2], 1)))
        x7 = self.relu(self.e_conv7(torch.cat([x6, x1], 1)))
        x8 = self.relu(self.e_conv8(x7))
        x9 = self.relu(self.e_conv9(x8))
        x10 = self.relu(self.e_conv10(torch.cat([x9, x5], 1)))
        x11 = self.relu(self.e_conv11(x10))
        attmap = self.sig(self.e_conv12(x11))

        return attmap[:, :, 4:-4, 4:-4]
    

class E_Net(nn.Module):

    def __init__(self):
        super(E_Net, self).__init__()

        self.relu = nn.LeakyReLU(inplace=True)
        self.sig = nn.Sigmoid()

        # Number of channels in the middle hidden layer
        number_f = 16

        # attmap estimate net
        self.e_conv1 = nn.Conv2d(1, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)                  
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv8 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv9 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv10 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv11 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv12 = nn.Conv2d(number_f, 1, 3, 1, 1, bias=True)  

        self.bn1 = nn.BatchNorm2d(number_f) 
        self.bn2 = nn.BatchNorm2d(number_f) 
        self.bn3 = nn.BatchNorm2d(number_f) 
        self.bn4 = nn.BatchNorm2d(number_f) 
        self.bn5 = nn.BatchNorm2d(number_f) 
        self.bn6 = nn.BatchNorm2d(number_f) 
        self.bn7 = nn.BatchNorm2d(number_f) 
        self.bn8 = nn.BatchNorm2d(number_f) 
        self.bn9 = nn.BatchNorm2d(number_f) 
        self.bn10 = nn.BatchNorm2d(number_f) 
        self.bn11 = nn.BatchNorm2d(number_f) 
        

    def forward(self, dark_image_V):
        
        dark_V = nn.ReflectionPad2d(16)(dark_image_V)

        # attmap estimate net
        x1 = self.relu(self.bn1(self.e_conv1(dark_V)))
        x2 = self.relu(self.bn2(self.e_conv2(x1)))
        x3 = self.relu(self.bn3(self.e_conv3(x2)))
        x4 = self.relu(self.bn4(self.e_conv4(x3)))
        x5 = self.relu(self.bn5(self.e_conv5(torch.cat([x4, x3], 1))))
        x6 = self.relu(self.bn6(self.e_conv6(torch.cat([x5, x2], 1))))
        x7 = self.relu(self.bn7(self.e_conv7(torch.cat([x6, x1], 1))))
        x8 = self.relu(self.bn8(self.e_conv8(x7)))
        x9 = self.relu(self.bn9(self.e_conv9(x8)))
        x10 = self.relu(self.bn10(self.e_conv10(torch.cat([x9, x5], 1))))
        x11 = self.relu(self.bn11(self.e_conv11(x10)))
        attmap = self.sig(self.e_conv12(x11))

        return attmap[:, :, 16:-16, 16:-16]
    

class C_Net(nn.Module):

    def __init__(self):
        super(C_Net, self).__init__()

        self.relu = nn.LeakyReLU(inplace=True)
        self.sig = nn.Sigmoid()

        # Number of channels in the middle hidden layer
        number_f = 16

        # attmap estimate net
        self.e_conv1 = nn.Conv2d(5, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)                  
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv8 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv9 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv10 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv11 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv12 = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)   
        

    def forward(self, x):

        x = nn.ReflectionPad2d(4)(x)

        # attmap estimate net
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x4, x3], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x5, x2], 1)))
        x7 = self.relu(self.e_conv7(torch.cat([x6, x1], 1)))
        x8 = self.relu(self.e_conv8(x7))
        x9 = self.relu(self.e_conv9(x8))
        x10 = self.relu(self.e_conv10(torch.cat([x9, x5], 1)))
        x11 = self.relu(self.e_conv11(x10))
        logit = self.sig(self.e_conv12(x11))

        return logit[:, :, 4:-4, 4:-4]


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm=False, leaky=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=True):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() if act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm=True, leaky=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, norm=norm, leaky=leaky)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm=True, leaky=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm, leaky=leaky)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm, leaky=leaky)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DenoiseNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(DenoiseNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = DoubleConv(in_channels, 32, norm=norm)
        self.down1 = Down(32, 64, norm=norm)
        self.down2 = Down(64, 128, norm=norm)
        self.down3 = Down(128, 256, norm=norm)

        self.up1 = Up(256, 128, bilinear=False, norm=norm)
        self.up2 = Up(128, 64, bilinear=False, norm=norm)
        self.up3 = Up(64, 32, bilinear=False, norm=norm)
        self.outc = OutConv(32, out_channels, act=True)

    def forward(self, x):

        x = nn.ReflectionPad2d(128)(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits[:, :, 128:-128, 128:-128]
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class ColorNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, norm=True):
        super(ColorNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = DoubleConv(in_channels, 32, norm=norm)
        self.down1 = Down(32, 64, norm=norm)
        self.down2 = Down(64, 128, norm=norm)
        self.down3 = Down(128, 256, norm=norm)

        self.up1 = Up(256, 128, bilinear=False, norm=norm)
        self.up2 = Up(128, 64, bilinear=False, norm=norm)
        self.up3 = Up(64, 32, bilinear=False, norm=norm)
        self.outc = OutConv(32, out_channels, act=True)

    def forward(self, x):
         
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
