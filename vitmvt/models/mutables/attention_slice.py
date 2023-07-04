import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MUTABLES
from .base_mutable import SliceOp
from .linear_slice import LinearSlice, LinearSlice_byhead
from .qkv_slice import QKV_Super, QkvSlice, QkvSlice_ratio


@MUTABLES.register_module()
class AttentionSlice(SliceOp):
    """Implements one encoder layer in Vision Transformer.

    Args:
        super_embed_dims (int): The feature dimension of supernet.
        num_heads (int): Parallel attention heads.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        attn_drop (float): The drop out rate for attention output weights.
            Defaults to 0.
        proj_drop (float): Dropout ratio of output. Defaults to 0.
        relative_position: use position encoding or not.
        max_relative_position: The max relative position distance.
        scale: Override default qk scale of ``head_dim ** -0.5``.
        change_qkv: search qkv or not.
    """

    def __init__(self,
                 super_embed_dim,
                 num_heads,
                 window_size=(14, 14),
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 scale=False,
                 change_qkv=True,
                 use_window_att=False,
                 unit=64):
        super().__init__()
        self.num_heads = num_heads
        self.super_embed_dim = super_embed_dim
        self.fc_scale = scale
        self.change_qkv = change_qkv
        self.use_window_att = use_window_att
        if change_qkv:
            self.qkv = QkvSlice_ratio(
                super_embed_dim, self.num_heads, unit=unit, bias=qkv_bias)
        else:
            self.qkv = LinearSlice(
                super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)

        if self.use_window_att:
            # used in detection
            self.window_size = (14, 14)
        else:
            self.window_size = window_size

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        self.get_value(num_heads,
                                       kind='max')))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.proj = LinearSlice_byhead(
            super_embed_dim,
            self.num_heads,
            unit=unit,
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        from mmcv.cnn.bricks.transformer import build_dropout
        dropout_layer = dict(type='DropPath', drop_prob=0.1)
        self.drop = build_dropout(dropout_layer)

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

    def init_weights(self):
        from mmcv.cnn.utils.weight_init import trunc_normal_
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward_inner(self, x, Hp, Wp):
        num_heads = self.get_value(self.num_heads)
        self.scale = 0.125
        B, N, C = x.shape

        if self.use_window_att:
            H, W = Hp, Wp
            x = x.reshape(B, H, W, C)
            pad_l = pad_t = 0
            pad_r = (self.window_size[1] -
                     W % self.window_size[1]) % self.window_size[1]
            pad_b = (self.window_size[0] -
                     H % self.window_size[0]) % self.window_size[0]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape

            x = window_partition(
                x, self.window_size[0])  # nW*B, window_size, window_size, C
            x = x.view(-1, self.window_size[1] * self.window_size[0],
                       C)  # nW*B, window_size*window_size, C
            B_w = x.shape[0]
            N_w = x.shape[1]

            qkv = self.qkv(x).reshape(B_w, N_w, 3, num_heads,
                                      -1).permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, num_heads,
                                      -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias[..., :num_heads]
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.use_window_att:
            x = (attn @ v).transpose(1, 2).reshape(B_w, N_w, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
            x = window_reverse(x, self.window_size[0], Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()
            x = x.view(B, H * W, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)

        x = self.drop(x)
        return x

    def export(self, **kwargs):
        return self


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
