"""
xz & 2024/5/1 15:50
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from networks.DSConv_pro import DSConv
from functools import partial
from timm.models.layers import SqueezeExcite,CondConv2d
from mmcv.cnn import ConvModule
from einops import rearrange
import numbers

inplace = True

nonlinearity = partial(F.relu, inplace=True)

class SSCNet(nn.Module):   #StratoSyncCrossNet
    def __init__(self, num_classes=1):
        super(SSCNet, self).__init__()
        # 1024*3
        filters = [48, 64, 160, 1280]
        model = models.efficientnet_v2_s(pretrained=True)
        # 1024*3
        self.downsampling1 = model.features[0]  # 512 512 24
        self.downsampling2 = model.features[1]  # 512 512 24

        #Encoder
        self.encoder1 = model.features[2]  # 256 256 48

        self.encoder2 = model.features[3]  # 128 128 64

        self.encoder31 = model.features[4]  # 64 64 128
        self.encoder32 = model.features[5]  # 64 64 160

        self.encoder41 = model.features[6]  # 32 32 256
        self.encoder42 = model.features[7]  # 32 32 1280

        #mddm
        self.conv1313 = ConvModule(24, 24, 13, 2, 6, groups=24)
        self.conv11 = nn.Conv2d(24, 48, 1)
        self.drdm1 = DRDM(1, filters[0])

        self.conv99 = ConvModule(filters[0], filters[0], 9, 2, 4, groups=filters[0])
        self.conv12 = nn.Conv2d(48, 64, 1)
        self.drdm2 = DRDM(2, filters[1])

        #att
        self.conv55 = ConvModule(filters[1], filters[1], 5, 2, 2, groups=filters[1])
        self.conv13 = nn.Conv2d(filters[1], filters[2], 1)
        self.att3 = CCAA(num=1, channels=filters[2], num_heads=8)

        self.conv33 = ConvModule(filters[2], filters[2], 3, 2, 1, groups=filters[2])
        self.conv14 = nn.Conv2d(filters[2], filters[3], 1)
        self.att4 = CCAA(num=2, channels=filters[3], num_heads=32)

        # MGIfM
        self.mgii = MGII()

        #Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)  # 1024*32
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)  # 1024*1

    def forward(self, x):
        # Encoder
        x = self.downsampling1(x)  # 512 512 24
        x = self.downsampling2(x)  # 512 512 24

        e1 = self.encoder1(x)  # 256 256 48

        e2 = self.encoder2(e1)  # 128 128 64

        e30 = self.encoder31(e2)  # 64 64 128
        e3 = self.encoder32(e30)  # 64 64 160

        e40 = self.encoder41(e3)  # 32 32 256
        e4 = self.encoder42(e40)  # 32 32 1280

        # DRDM
        e11 = self.conv1313(x)  # 256 256 24
        e1att = self.conv11(e11)  # 256 256 48
        e1 = e1att + e1  # 256 256 48
        drdm1 = self.drdm1(e1)  # 256 256 48

        e21 = self.conv99(drdm1)  # 128 128 48
        e2att = self.conv12(e21)  # 128 128 64
        e2 = e2att + e2  # 128 128 64
        drdm2 = self.drdm2(e2)  # 128 128 64

        # att
        e31 = self.conv55(drdm2)
        e3att = self.conv13(e31)
        att3 = self.att3(e3, e3att)

        e41 = self.conv33(att3)
        e4att = self.conv14(e41)
        att4 = self.att4(e4, e4att)

        #Center
        mgii = self.mgii(e4, att3, att4)

        # Decoder
        d3 = self.decoder4(att4+mgii)# 32 32 1280 => # 64 64 160 #e4要改
        d2 = self.decoder3(att3+d3) # 128 128 64
        d1 = self.decoder2(drdm2+d2)# 256 256 48
        d0 = self.decoder1(drdm1+d1) #512 512 32

        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)



class DRDM(nn.Module):  #Dynamic Road Detail Matcher
    def __init__(self, num: int, channels: int):
        super(DRDM, self).__init__()
        self.num = num

        self.conv11 = ConvModule(channels, channels, (1, 19 - 4 * num), 1, (0, 9 - 2 * num), groups=channels)
        self.conv12 = ConvModule(channels, channels, (1, 17 - 4 * num), 1, (0, 8 - 2 * num), groups=channels)
        self.dsconv13 = DSConv(channels, channels, kernel_size=15 - 4 * num, extend_scope=1,
                               morph=0, if_offset=True, device="cuda")

        self.conv21 = ConvModule(channels, channels, (19 - 4 * num, 1), 1, (9 - 2 * num, 0), groups=channels)
        self.conv22 = ConvModule(channels, channels, (17 - 4 * num, 1), 1, (8 - 2 * num, 0), groups=channels)
        self.dsconv23 = DSConv(channels, channels, kernel_size=15 - 4 * num, extend_scope=1,
                               morph=1, if_offset=True, device="cuda")

        self.conv31 = ConvModule(channels, channels, 19 - 4 * num, 1, 9 - 2 * num, groups=channels)
        self.conv32 = ConvModule(channels, channels, 17 - 4 * num, 1, 8 - 2 * num, groups=channels)
        self.conv33 = ConvModule(channels, channels, 15 - 4 * num, 1, 7 - 2 * num, groups=channels)

        self.dwconv11 = ConvModule(channels, channels, 1, 1, 0, groups=channels)
        self.bnnorm = nn.BatchNorm2d(channels)

        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.dsconv13(self.conv12(self.conv11(x)))
        x2 = self.dsconv23(self.conv22(self.conv21(x)))
        x3 = self.conv33(self.conv32(self.conv31(x)))

        x0 = x + x1 + x2 + x3
        x0 = nonlinearity(self.bnnorm(x0))

        x = self.gelu(x + x0)
        return x


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)



class CCAA(nn.Module): #CrossContextualAdaptiveAttention
    def __init__(self, num, channels, num_heads, LayerNorm_type = 'WithBias',):
        super(CCAA, self).__init__()
        self.num = num
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm1 = LayerNorm(channels, LayerNorm_type)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

        self.convputong = nn.Conv2d(channels, channels, kernel_size= 7 - 2 * num, padding=3 - num)
        self.dconv = nn.Conv2d(channels, channels, kernel_size=9 - 2 * num, dilation=2, padding=8 - 2 * num)
        self.dwconv = ConvModule(channels, channels, (11 - 2 * num, 11 - 2 * num), 1,
                                  (5 - num, 5 - num), groups=channels,
                                  norm_cfg=None, act_cfg=None)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        x11 = self.convputong(x1) + x1
        x12 = self.dconv(x11) + x11
        x1c = self.dwconv(x12) + x12

        x21 = self.convputong(x2) + x2
        x22 = self.dconv(x21) + x21
        x2c = self.dwconv(x22) + x12

        out1 = self.project_out(x1c)
        out2 = self.project_out(x2c)
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x = self.project_out(out3) + self.project_out(out4) + x1 + x2

        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class MGII(nn.Module): #Comprehensive Multi-Scale Global Integrator
    def __init__(self):
        super(MGII, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.block = RepViTBlock(1280, 2560, 1280, 3, 1, True, True)
        self.conv13 = nn.Conv2d(160, 1280, kernel_size=1)

        self.convdown = nn.Conv2d(5120, 1280, kernel_size=1)
        self.norm = nn.BatchNorm2d(1280)

    def forward(self, x, x3, x4):
        x3 = self.conv13(x3)
        x3 = self.gap(x3)
        x3 = torch.mul(x, x3)
        x3 = x + x3
        x3 = self.block(x3)

        x4 = self.gap(x4)
        x4 = torch.mul(x, x4)
        x4 = x + x4
        x4 = self.block(x4)

        xx = self.gap(x)
        xx = torch.mul(x, xx)
        xx = x + xx
        xx = self.block(xx)

        x = torch.cat((x, x3, x4, xx), 1)
        x = self.convdown(x)
        x = nonlinearity(self.norm(x))

        return x


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


