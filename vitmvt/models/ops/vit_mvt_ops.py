import torch
import torch.nn as nn

import mmcv
from mmcv.cnn import ConvModule
from mmcv.cnn import build_activation_layer, build_norm_layer, build_conv_layer

from ..mutables import Conv2dSlice

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        out += identity
        out = self.relu(out)
        out = out.flatten(2).transpose(1, 2)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, self.planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            self.inplanes,
            self.planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, self.planes, self.planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.dilation = dilation

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.relu(out)
        out = out.flatten(2).transpose(1, 2)
        return out


class Channel_interaction(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        divisor (int): Divisor for make_divisible. Default: 1.
        use_avgpool (bool): Use AdaptiveAvgPool or not. Default: False.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 channels,
                 dynamic_channels=None,
                 ratio=16,
                 conv_cfg=None,
                 divisor=1,
                 use_avgpool=True,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super(Channel_interaction, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1) if use_avgpool else None
        if dynamic_channels:
            dynamic_channels = dynamic_channels
        else:
            dynamic_channels = channels // ratio

        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=dynamic_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=dynamic_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x, use_multipy=True):
        if self.global_avgpool is not None:
            out = self.global_avgpool(x)
        else:
            out = x.mean(3, keepdim=True).mean(2, keepdim=True)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out if use_multipy else out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = Conv2dSlice(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x) + x
        x = x.flatten(2).transpose(1, 2)
        return x
