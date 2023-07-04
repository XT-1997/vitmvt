import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.utils import to_2tuple

from ..builder import MUTABLES
from .base_mutable import SliceOp


class Patchembed(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=576,
                 scale=False,
                 sampled_scale=1.0,
                 conv_cfg=None):
        super(Patchembed, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        kwargs = dict(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size)
        self.proj = build_conv_layer(cfg=conv_cfg, **kwargs)
        self.super_embed_dim = embed_dim

        self.scale = scale
        self.sampled_scale = sampled_scale

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return (x, Hp, Wp)


@MUTABLES.register_module()
class PatchembedSlice(SliceOp):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=None,
                 scale=False,
                 sampled_scale=1.0,
                 conv_cfg=None):
        super(PatchembedSlice, self).__init__()

        self.embedding_dim = embed_dim
        max_embed_dim = self.get_value(self.embedding_dim, kind='max')
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size // 2)
        num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        kwargs = dict(
            in_channels=64,
            out_channels=max_embed_dim,
            kernel_size=patch_size,
            stride=patch_size)

        self.proj = build_conv_layer(cfg=conv_cfg, **kwargs)
        self.super_embed_dim = max_embed_dim

        self.scale = scale
        self.sampled_scale = sampled_scale

    def forward_inner(self, x):
        sample_embed_dim = self.get_value(self.embedding_dim)
        sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        sampled_bias = self.proj.bias[:sample_embed_dim, ...]
        x = F.conv2d(
            x,
            sampled_weight,
            sampled_bias,
            stride=self.patch_size,
            padding=self.proj.padding,
            dilation=self.proj.dilation)

        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        if self.scale:
            self.sampled_scale = self.embedding_dim / sample_embed_dim
            x = x * self.sampled_scale
        return (x, Hp, Wp)

    def forward(self, *args, **kwargs):
        """For multiple inheritance, we use the SliceOp's ``forward`` method to
        proxy.

        Note that, each mutiple inheritance mutable should realize this.
        """
        return SliceOp.forward(self, *args, **kwargs)

    def export(self, **kwargs):
        """Export LayerNormSlice to nn.LayerNorm."""

        embedding_dim = kwargs.get('embed_channels.0',
                                   self.get_value(self.embedding_dim))

        if self.scale:
            self.sampled_scale = self.embedding_dim / embedding_dim

        export_module = Patchembed(
            img_size=224,
            patch_size=8,
            in_chans=64,
            embed_dim=embedding_dim,
            scale=self.scale,
            sampled_scale=self.sampled_scale,
            conv_cfg=dict(type='Conv2d'))
        weight, bias = self.proj.slice_params(64, embedding_dim, (8, 8), 1)

        export_module.proj.weight.data.copy_(weight)
        export_module.proj.bias.data.copy_(bias)

        return export_module