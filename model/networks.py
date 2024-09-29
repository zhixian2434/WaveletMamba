import torch.nn as nn
import torch
import torch.nn.functional as F
from mamba_ssm import Mamba
import warnings
import math


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class cross_attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)
    """
    x: b,c,h,w
    y: b,c,2h,2w

    q:b,h,c,w
    k:b,2h,c,2w
    v:b,2h,c,2w

    qk:b
    """

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Dilated_Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_Resblock, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1))

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out


class CGRM(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(CGRM, self).__init__()

        self.conv_head1 = Depth_conv(in_channels, out_channels)
        self.conv_head2 = Depth_conv(in_channels, out_channels)
        self.conv_head3 = Depth_conv(in_channels, out_channels)
        self.conv_head4 = Depth_conv(in_channels, out_channels)

        self.dilated_block_LH = Dilated_Resblock(out_channels, out_channels)
        self.dilated_block_HL = Dilated_Resblock(out_channels, out_channels)

        self.cross_attention0 = cross_attention(out_channels, num_heads=8)
        self.dilated_block_HH = Dilated_Resblock(out_channels, out_channels)
        self.cross_attention1 = cross_attention(out_channels, num_heads=8)
        self.cross_attention2 = cross_attention(out_channels, num_heads=8)

        self.conv_tail = Depth_conv(out_channels, in_channels)

    def forward(self, x, y):
        b, c, h, w = x.shape
        residual = x

        x_HL, x_LH, x_HH = x[:b//3, ...], x[b//3:2*b//3, ...], x[2*b//3:, ...]

        x_HL = self.conv_head1(x_HL)
        x_LH = self.conv_head2(x_LH)
        x_HH = self.conv_head3(x_HH)
        x_A = self.conv_head4(y)

        x_A_LH = self.cross_attention0(x_A, x_LH)
        x_A_HL = self.cross_attention1(x_A, x_HL)
        x_A_HH = self.cross_attention2(x_A, x_HH)

        x_HL = self.dilated_block_HL(x_A_HL)
        x_LH = self.dilated_block_LH(x_A_LH)
        x_HH = self.dilated_block_HH(x_A_HH)

        out = self.conv_tail(torch.cat((x_HL, x_LH, x_HH), dim=0))

        return out + residual


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x_in):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        x_mamba = self.mamba(x)
        out = x_mamba.reshape(b, h, w, c)
        return out
    

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
    

class FeedForward(nn.Module):
    def __init__(self, in_channel, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(in_channel * mult, in_channel * mult, 3, 1, 1,
                      bias=False, groups=in_channel * mult),
            GELU(),
            nn.Conv2d(in_channel * mult, in_channel, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MambaBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel

        self.ln1 = nn.LayerNorm(in_channel)
        self.Mamba = MambaLayer(dim=self.in_channel, num_slices=16)
        self.ln2 = nn.LayerNorm(in_channel)
        self.ff = FeedForward(self.in_channel)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.Mamba(self.ln1(x)) + x
        x = self.ff(self.ln2(x)) + x
        out = x.permute(0, 3, 1, 2)
        return out


class ConvAttBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_feat):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_feat = n_feat

        self.conv1 = nn.Conv2d(self.in_channel, self.n_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(self.n_feat + self.n_feat, self.out_channel, 1, 1, bias=True)
        self.MambaBlock = MambaBlock(self.n_feat)

        self.conv_block = Dilated_Resblock(n_feat, n_feat)
    
    def forward(self, x):
        fea = self.conv1(x)
        conv_x = self.conv_block(fea) 
        mamba_x = self.MambaBlock(fea) 
        out = self.conv2(torch.cat([conv_x, mamba_x], 1))

        return out
    

class WMambaUNet(nn.Module): # Wavelet-based Vision Mamba Model 

    def __init__(self, in_channel=3, out_channel=3, n_feat=32):
        super().__init__()
        self.head = [Depth_conv(in_channel, n_feat)] 
        self.convatt1 = [ConvAttBlock(in_channel=n_feat, out_channel=n_feat, n_feat=n_feat)]
        self.down1 = [nn.Conv2d(n_feat, n_feat*2, 2, 2)] #B x 2*C x H/2 x W/2
        self.convatt2 = [ConvAttBlock(in_channel=n_feat*2, out_channel=n_feat*2, n_feat=n_feat*2)]
        self.down2 = [nn.Conv2d(n_feat*2, n_feat*4, 2, 2)] #B x 4*C x H/4 x W/4
        self.convatt3 = [ConvAttBlock(in_channel=n_feat*4, out_channel=n_feat*4, n_feat=n_feat*4)] #B x 4*C x H/4 x W/4
        self.down3 = [nn.Conv2d(n_feat*4, n_feat*8, 2, 2)] #B x 8*C x H/8 x W/8
        self.body = [ConvAttBlock(in_channel=n_feat*8, out_channel=n_feat*8, n_feat=n_feat*8)] #B x 8*C x H/8 x W/8
        self.up1 = [nn.ConvTranspose2d(n_feat*8, n_feat*4, 2, 2, 0, bias=False)] #B x 4*C x H/4 x W/4
        self.convatt4 = [ConvAttBlock(in_channel=n_feat*4, out_channel=n_feat*4, n_feat=n_feat*4)]
        self.up2 = [nn.ConvTranspose2d(n_feat*4, n_feat*2, 2, 2, 0, bias=False)] #B x 2*C x H/2 x W/2
        self.convatt5 = [ConvAttBlock(in_channel=n_feat*2, out_channel=n_feat*2, n_feat=n_feat*2)]
        self.up3 = [nn.ConvTranspose2d(n_feat*2, n_feat, 2, 2, 0, bias=False)] #B x C x H x W
        self.convatt6 = [ConvAttBlock(in_channel=n_feat, out_channel=n_feat, n_feat=n_feat)]
        self.tail = [Depth_conv(n_feat, out_channel)] 

        self.conv1 = nn.Conv2d(n_feat*8, n_feat*4, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(n_feat*4, n_feat*2, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(n_feat*2, n_feat*1, 1, 1, bias=False)

        self.conv_block1 = Dilated_Resblock(n_feat*4, n_feat*4)
        self.conv_block2 = Dilated_Resblock(n_feat*2, n_feat*2)
        self.conv_block3 = Dilated_Resblock(n_feat, n_feat)

        self.m_head = nn.Sequential(*self.head)
        self.m_convatt1 = nn.Sequential(*self.convatt1)
        self.m_down1 = nn.Sequential(*self.down1)
        self.m_convatt2 = nn.Sequential(*self.convatt2)
        self.m_down2 = nn.Sequential(*self.down2)
        self.m_convatt3 = nn.Sequential(*self.convatt3)
        self.m_down3 = nn.Sequential(*self.down3)
        self.m_body = nn.Sequential(*self.body)
        self.m_up1 = nn.Sequential(*self.up1)
        self.m_convatt4 = nn.Sequential(*self.convatt4)
        self.m_up2 = nn.Sequential(*self.up2)
        self.m_convatt5 = nn.Sequential(*self.convatt5)
        self.m_up3 = nn.Sequential(*self.up3)
        self.m_convatt6 = nn.Sequential(*self.convatt6)
        self.m_tail = nn.Sequential(*self.tail)

    
    def forward(self, x):

        x1 = self.m_head(x)
        x2 = self.m_convatt1(x1)
        y1 = self.conv_block3(x2)
        x3 = self.m_down1(x2)
        x4 = self.m_convatt2(x3)
        y2 = self.conv_block2(x4)
        x5 = self.m_down2(x4)
        x6 = self.m_convatt3(x5)
        y3 = self.conv_block1(x6)
        x7 = self.m_down3(x6)
        x8 = self.m_body(x7)
        x9 = self.m_up1(x8)
        fusion1 = self.conv1(torch.cat([x9, y3], 1))
        x10 = self.m_convatt4(fusion1)
        x11 = self.m_up2(x10)
        fusion2 = self.conv2(torch.cat([x11, y2], 1))
        x12 = self.m_convatt5(fusion2)
        x13 = self.m_up3(x12)
        fusion3 = self.conv3(torch.cat([x13, y1], 1))
        x14 = self.m_convatt6(fusion3)
        out = self.m_tail(x14)

        return out
    

class WMMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Mamba = WMambaUNet()
        self.CGRM = CGRM()

    def forward(self, a_in, h_in):
        restore_A = self.Mamba(a_in)
        restore_H = self.CGRM(h_in, a_in)

        return restore_A, restore_H
    