from turtle import forward
import torch
import torch.nn as nn
from torchvision import models
from ms_ssim import *
import numpy as np
import torchvision
from pytorch_msssim import SSIM, MS_SSIM


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, x1, x2):
        x1 = x1 - x2
        x1 = abs(x1)
        return x1.mean()
    

def laplace(img):
    laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype='float32')
    laplace_kernel = laplace_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(laplace_kernel)).cuda()
    edge_detect = F.conv2d(Variable(img), weight, padding=1)
    edge_detect = edge_detect.detach()
    return edge_detect


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, x1, x2):
        x1 = x1 - x2
        x1 = x1.pow(2)
        return x1.mean()
    

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []

        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i+1) in indices:
                out.append(X)
        
        return out


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss1(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss1, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8]
        self.indices = indices or [2, 7]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True)
        else:
            self.normalize = None
        print("Vgg: Weights: ", self.weights, " indices: ", self.indices, " normalize: ", self.normalize)

    def __call__(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):

        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = (torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)).sum()
        w_tv = (torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    

class WTVLoss2(nn.Module):
    def __init__(self):
        super(WTVLoss2, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, data, aux):
        N, C, H, W = data.shape
        
        data_dw = F.pad(torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:]), (1, 0, 0, 0))
        data_dh = F.pad(torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :]), (0, 0, 1, 0))
        aux_dw = F.pad(torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:]), (1, 0, 0, 0))
        aux_dh = F.pad(torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :]), (0, 0, 1, 0))

        data_d = data_dw + data_dh
        aux_d = aux_dw + aux_dh
        loss2 = torch.norm(data_d / (aux_d + 1e-2)) / (C * H * W)
        return loss2
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class wing_loss(nn.Module):
    def __init__(self, w=3000.0, epsilon=0.5):
        super(wing_loss, self).__init__()
        self.w = w
        self.epsilon = epsilon

    def forward(self, origin, enhance):
        origin_gradmap = origin
        enhance_gradmap = enhance
        difference = origin_gradmap - enhance_gradmap
        l = self.w * torch.log(1.0 + difference/self.epsilon)
        zero = torch.zeros_like(l, requires_grad=True)
        t = torch.where(torch.le(difference, zero), zero, l)
        loss = torch.mean(t)

        return loss


class L_TV_weighted(nn.Module):
    def __init__(self, TVLoss_weight=0.0001):
        super(L_TV_weighted, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x, gt):
        rgb_grad = gt
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        grad_h = torch.abs(rgb_grad[:, :, 1:, :])+0.000001
        grad_w = torch.abs(rgb_grad[:, :, :, 1:])+0.000001
        h_tv = (torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)*(1/grad_h)).sum()
        w_tv = (torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)*(1/grad_w)).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size
        

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    

class WTVLoss(torch.nn.Module):
    def __init__(self):
        super(WTVLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, data, aux):
        N, C, H, W = data.shape
        
        data_dw = F.pad(torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:]), (1, 0, 0, 0))
        data_dh = F.pad(torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :]), (0, 0, 1, 0))
        aux_dw = F.pad(torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:]), (1, 0, 0, 0))
        aux_dh = F.pad(torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :]), (0, 0, 1, 0))

        data_d = data_dw + data_dh
        aux_d = aux_dw + aux_dh
        loss1 = self.criterion(data_d, aux_d)
        loss2 = torch.norm(data_d / (aux_d + 1e-2)) / (C * H * W)
        return loss2
    

#Calculated gradient loss
class GradientLoss(nn.Module):
    def __init__(self, channel):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.gradient = Gradient_Net()
        self.channel = channel
    def compute_gradient(self, img, channel):
        gradimg = self.gradient(img, channel)
        return gradimg

    def forward(self, predict, target):
        predict_grad = self.compute_gradient(predict, channel=self.channel)
        target_grad = self.compute_gradient(target, channel=self.channel)

        return self.loss(predict_grad, target_grad)


class Gradient_Net(nn.Module):
    def __init__(self):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x, channel):
        b, c, h, w = x.shape
        if channel == 3:
            x = (0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]).reshape(b, 1, h, w)
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient
    

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
    

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = [h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3]
        return out


class VGG_LOSS(nn.Module):
    def __init__(self):
        super(VGG_LOSS, self).__init__()
        self.vgg = VGG16()
        self.l2 = MSE()

    def forward(self, dark, gth):
        output_features = self.vgg(dark)
        gth_features = self.vgg(gth)
        out_1 = self.l2(output_features[0], gth_features[0])
        out_2 = self.l2(output_features[1], gth_features[1])
        out_3 = self.l2(output_features[2], gth_features[2])
        out_4 = self.l2(output_features[3], gth_features[3])
        return (out_1 + out_2 + out_3 + out_4) / 4
    

class color_loss(nn.Module):
    def __init__(self):
        super(color_loss, self).__init__()

    def forward(self, x, y):
        b, c, h, w = x.shape

        mr_x, mg_x, mb_x = torch.split(x, 1, dim=1)
        mr_x, mg_x, mb_x = mr_x.reshape([b, 1, -1, 1]), mg_x.reshape([b, 1, -1, 1]), mb_x.reshape([b, 1, -1, 1])
        xx = torch.cat([mr_x, mg_x, mb_x], dim=3).squeeze(1) + 0.000001

        mr_y, mg_y, mb_y = torch.split(y, 1, dim=1)
        mr_y, mg_y, mb_y = mr_y.reshape([b, 1, -1, 1]), mg_y.reshape([b, 1, -1, 1]), mb_y.reshape([b, 1, -1, 1])
        yy = torch.cat([mr_y, mg_y, mb_y], dim=3).squeeze(1) + 0.000001

        xx = xx.reshape(h * w * b, 3)
        yy = yy.reshape(h * w * b, 3)
        l_x = torch.sqrt(pow(xx[:, 0], 2) + pow(xx[:, 1], 2) + pow(xx[:, 2], 2))
        l_y = torch.sqrt(pow(yy[:, 0], 2) + pow(yy[:, 1], 2) + pow(yy[:, 2], 2))
        xy = xx[:, 0] * yy[:, 0] + xx[:, 1] * yy[:, 1] + xx[:, 2] * yy[:, 2]
        cos_angle = xy / (l_x * l_y + 0.000001)
        angle = torch.acos(cos_angle.cpu())
        angle2 = angle * 360 / 2 / np.pi
        # re = angle2.reshape(b, -1)
        an_mean = torch.mean(angle2) / 100
        return an_mean.cuda()


class L_grad_cosist(nn.Module):

    def __init__(self):
        super(L_grad_cosist, self).__init__()
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

    def gradient_of_one_channel(self,x,y):
        D_org_right = F.conv2d(x , self.weight_right, padding=1)
        D_org_down = F.conv2d(x , self.weight_down, padding=1)
        D_enhance_right = F.conv2d(y , self.weight_right, padding=1)
        D_enhance_down = F.conv2d(y , self.weight_down, padding=1)
        return torch.abs(D_org_right),torch.abs(D_enhance_right),torch.abs(D_org_down),torch.abs(D_enhance_down)

    def gradient_Consistency_loss_patch(self,x,y):
        # B*C*H*W
        min_x = torch.abs(x.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        #B*1*1,3
        product_separte_color = (x*y).mean([2,3],keepdim=True)
        x_abs = (x**2).mean([2,3],keepdim=True)**0.5
        y_abs = (y**2).mean([2,3],keepdim=True)**0.5
        loss1 = (1-product_separte_color/(x_abs*y_abs+0.00001)).mean() + torch.mean(torch.acos(product_separte_color/(x_abs*y_abs+0.00001)))

        product_combine_color = torch.mean(product_separte_color,1,keepdim=True)
        x_abs2 = torch.mean(x_abs**2,1,keepdim=True)**0.5
        y_abs2 = torch.mean(y_abs**2,1,keepdim=True)**0.5
        loss2 = torch.mean(1-product_combine_color/(x_abs2*y_abs2+0.00001)) + torch.mean(torch.acos(product_combine_color/(x_abs2*y_abs2+0.00001)))
        return loss1 + loss2


    def forward(self, x, y):

        x_R1,y_R1, x_R2,y_R2  = self.gradient_of_one_channel(x[:,0:1,:,:],y[:,0:1,:,:])
        x_G1,y_G1, x_G2,y_G2  = self.gradient_of_one_channel(x[:,1:2,:,:],y[:,1:2,:,:])
        x_B1,y_B1, x_B2,y_B2  = self.gradient_of_one_channel(x[:,2:3,:,:],y[:,2:3,:,:])
        x = torch.cat([x_R1,x_G1,x_B1,x_R2,x_G2,x_B2],1)
        y = torch.cat([y_R1,y_G1,y_B1,y_R2,y_G2,y_B2],1)

        B,C,H,W = x.shape
        loss = self.gradient_Consistency_loss_patch(x,y)
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,0:W//2],y[:,:,0:H//2,0:W//2])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,0:W//2],y[:,:,H//2:,0:W//2])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,W//2:],y[:,:,0:H//2,W//2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,W//2:],y[:,:,H//2:,W//2:])

        return loss #+loss1#+torch.mean(torch.abs(x-y))#+loss1


class L_bright_cosist(nn.Module):

    def __init__(self):
        super(L_bright_cosist, self).__init__()

    def gradient_Consistency_loss_patch(self,x,y):
        # B*C*H*W
        min_x = torch.abs(x.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        #B*1*1,3
        product_separte_color = (x*y).mean([2,3],keepdim=True)
        x_abs = (x**2).mean([2,3],keepdim=True)**0.5
        y_abs = (y**2).mean([2,3],keepdim=True)**0.5
        loss1 = (1-product_separte_color/(x_abs*y_abs+0.00001)).mean() + torch.mean(torch.acos(product_separte_color/(x_abs*y_abs+0.00001)))

        product_combine_color = torch.mean(product_separte_color,1,keepdim=True)
        x_abs2 = torch.mean(x_abs**2,1,keepdim=True)**0.5
        y_abs2 = torch.mean(y_abs**2,1,keepdim=True)**0.5
        loss2 = torch.mean(1-product_combine_color/(x_abs2*y_abs2+0.00001)) + torch.mean(torch.acos(product_combine_color/(x_abs2*y_abs2+0.00001)))
        return loss1 + loss2

    def forward(self, x, y):
        
        loss = self.gradient_Consistency_loss_patch(x,y)

        return loss


class L_recon(nn.Module):

    def __init__(self):
        super(L_recon, self).__init__()
        self.ssim_loss = SSIM()

    def forward(self, R_low, high):
        L1 = torch.abs(R_low - high).mean()
        L2 = (1- self.ssim_loss(R_low,high)).mean()
        return L1 + L2