import copy
import math
import warnings

import torch
import torch.nn as nn
from vitmvt.space import build_search_space_recur
from vitmvt.models.ops import Bottleneck, BasicBlock, Channel_interaction, DWConv
from mmcv.cnn import build_activation_layer, build_norm_layer, build_conv_layer

from mmcv.utils import to_2tuple

from ..builder import BACKBONES
from ..mutables import (AttentionSlice, LinearRatioSlice, ParameterSlice,
                        PatchembedSlice, SequentialSlice)

try:
    from mmcls.models.backbones.base_backbone import BaseBackbone
except ImportError:
    from ...utils import get_placeholder
    BaseBackbone = get_placeholder('mmcls')


@BACKBONES.register_module()
class VIT_MVT_BACKBONE(BaseBackbone):
    """VIT_MVT_BACKBONE.

    Args:
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LayerNormSlice')``.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
        search_space (dict): Supernet's search_space.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 use_inc_mode=False,
                 qkv_bias=True,
                 out_indices=None,
                 conv_cfg=dict(type='Conv2dSlice'),
                 norm_cfg=dict(type='LayerNormSlice', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 final_norm=True,
                 init_cfg=None,
                 pretrained=None,
                 search_space=None,
                 use_window_att=False,
                 is_base=False):
        super(VIT_MVT_BACKBONE, self).__init__(init_cfg)
        if search_space:
            depth = search_space.get('depth', None)
            search_space['mlp_ratio'] = self.make_n_layers(
                search_space.get('mlp_ratio', None), max(depth['data']))
            search_space['num_heads'] = self.make_n_layers(
                search_space.get('num_heads', None), max(depth['data']))
            search_space['depth'] = search_space['depth']
            search_space['embed_channels'] = [search_space['embed_channels']]
            search_space = build_search_space_recur(search_space)
            depth = search_space['depth']
            mlp_ratio = search_space['mlp_ratio']
            num_heads = search_space['num_heads']
            embed_channels = search_space['embed_channels'][0]
            stage_settings = generate_supernet_settings(
                depth, mlp_ratio, num_heads)
        self.embed_dims = embed_channels
        self.depth = depth
        self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.window_size = (self.img_size[0] // patch_size,
                            self.img_size[1] // patch_size)
        self.patch_shape = self.window_size

        self.out_indices = out_indices
        self.use_window_att = use_window_att
        self.use_inc_mode = use_inc_mode
        self.is_base = is_base
        # Set patch embedding
        _patch_cfg = dict(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=self.embed_dims,
            scale=False,
            sampled_scale=1.0,
            conv_cfg=conv_cfg,
        )

        self.cnn_pre = True
        import mmcv
        rank, _ = mmcv.runner.get_dist_info()
        if self.cnn_pre:
            if rank < 32:
                self.cnn_pre = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
            else:
                self.cnn_pre = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.SyncBatchNorm(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.SyncBatchNorm(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.SyncBatchNorm(64),
                    nn.ReLU(inplace=True),
                )

        self.patch_embed = PatchembedSlice(**_patch_cfg)

        self.patch_resolution = [
            img_size // patch_size, img_size // patch_size
        ]
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set cls token
        self.use_cls_token = False
        if self.use_cls_token:
            self.pos_embed = ParameterSlice(1, num_patches + 1,
                                            self.embed_dims)
            self.cls_token = ParameterSlice(1, 1, self.embed_dims)
        else:
            self.pos_embed = ParameterSlice(1, num_patches, self.embed_dims)

        # stochastic depth decay rule
        self.layers = TotalLayerSlice(
            self.depth,
            self.embed_dims,
            norm_cfg=norm_cfg,
            qkv_bias=qkv_bias,
            window_size=self.window_size,
            stage_settings=stage_settings,
            use_window_att=self.use_window_att,
            is_base=self.is_base)

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.pretrained = pretrained
        self.interpolate_mode = 'bicubic'

    def init_weights(self, *args, **kwargs):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            from mmcv.runner import CheckpointLoader, load_state_dict
            from vitmvt.utils import get_root_logger
            logger = get_root_logger()
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            from collections import OrderedDict
            new_ckpt = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('model.backbone'):
                    new_key = k.replace('model.backbone.', '')
                    new_ckpt[new_key] = v

            state_dict = new_ckpt
            all_keys = list(state_dict.keys())
            for key in all_keys:
                
                if 'attn.proj.weight' in key:
                    new_key = key.replace('weight', 'specific_weight')
                    state_dict[new_key] = state_dict[key]

                if 'attn.proj.bias' in key:
                    new_key = key.replace('bias', 'specific_bias')
                    state_dict[new_key] = state_dict[key]

                if 'attn.qkv.weight' in key:
                    new_key = key.replace('weight', 'specific_weight')
                    state_dict[new_key] = state_dict[key]

                if 'attn.qkv.bias' in key:
                    new_key = key.replace('bias', 'specific_bias')
                    state_dict[new_key] = state_dict[key]

                if 'fc1.weight' in key or 'fc2.weight' in key:
                    new_key = key.replace('weight', 'specific_weight')
                    state_dict[new_key] = state_dict[key]
                
                if 'fc1.bias' in key or 'fc2.bias' in key:
                    new_key = key.replace('bias', 'specific_bias')
                    state_dict[new_key] = state_dict[key]
                
                # In order to keep the center of pos_bias as consistent as
                # possible after interpolation, and vice versa in the edge
                # area, the geometric sequence interpolation method is adopted.
                if 'relative_position_bias_table' in key:
                    rel_pos_bias = state_dict[key]
                    src_num_pos, num_attn_heads = rel_pos_bias.size()
                    dst_num_pos, _ = self.state_dict()[key].size()
                    dst_patch_shape = self.patch_shape
                    if dst_patch_shape[0] != dst_patch_shape[1]:
                        raise NotImplementedError()
                    if src_num_pos != dst_num_pos:
                        src_size = int(src_num_pos**0.5)
                        dst_size = int(dst_num_pos**0.5)
                        from mmcls.models.utils import \
                            resize_relative_position_bias_table
                        new_rel_pos_bias = resize_relative_position_bias_table(
                            src_size, dst_size,
                            rel_pos_bias, num_attn_heads)
                        logger.info('Resize the relative_position_bias_table from '
                                    f'{rel_pos_bias.shape} to '
                                    f'{new_rel_pos_bias.shape}')

                        state_dict[key] = new_rel_pos_bias

                        index_buffer = key.replace('bias_table', 'index')
                        del state_dict[index_buffer]

            if 'pos_embed.cls_token' in state_dict.keys():
                if self.pos_embed.cls_token.shape != state_dict[
                        'pos_embed.cls_token'].shape:
                    logger.info(
                        msg=f'Resize the pos_embed shape from '
                        f'{state_dict["pos_embed.cls_token"].shape} to '
                        f'{self.pos_embed.cls_token.shape}')
                    h, w = self.img_size
                    if self.use_cls_token:
                        pos_size = int(
                            math.sqrt(
                                state_dict['pos_embed.cls_token'].shape[1]))
                    else:
                        pos_size = int(
                            math.sqrt(
                                state_dict['pos_embed.cls_token'].shape[1] -
                                1))
                    state_dict['pos_embed.cls_token'] = self.resize_pos_embed(
                        state_dict['pos_embed.cls_token'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)
            load_state_dict(self, state_dict, strict=False, logger=logger)
        else:
            super(VIT_MVT_BACKBONE, self).init_weights()

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positiong embeding method.
        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            pos_h = self.img_size[0] // self.patch_size
            pos_w = self.img_size[1] // self.patch_size

            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return patched_img + pos_embed

    def resize_pos_embed(self, pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.
        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        # keep dim for easy deployment
        if self.use_cls_token:
            cls_token_weight = pos_embed[:, 0:1]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        from mmseg.ops import resize
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        if self.use_cls_token:
            pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        else:
            pos_embed = pos_embed_weight
        return pos_embed

    def forward(self, x, drop_ratio=0, drop_path_ratio=0):
        B = x.shape[0]
        if self.cnn_pre:
            x = self.cnn_pre(x)
        x, Hp, Wp = self.patch_embed(x)
        if self.use_cls_token:
            cls_tokens = self.cls_token().expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed()
        x = self._pos_embeding(x, (Hp, Wp), self.pos_embed())
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, Hp, Wp)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            # TODO: add out_indices for seg and det
            if self.out_indices and i in self.out_indices:
                if self.use_cls_token:
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, Hp, Wp, C).permute(0, 3, 1,
                                                        2).contiguous()
                outs.append(out)

        if self.out_indices is not None:
            return tuple(outs)
        else:
            if self.use_cls_token:
                return torch.mean(x[:, 1:], dim=1)
            else:
                return torch.mean(x, dim=1)

    def make_n_layers(self, type_dict, n):
        """copy the dict for n times and change the key name then generate a
        dict."""
        new_dict_list = list()
        type_dict['key'] = type_dict['key'].split('.')[0] + '.' + '0'
        new_dict_list.append(type_dict)
        for i in range(n - 1):
            new_dict = copy.deepcopy(type_dict)
            new_dict['key'] = new_dict['key'].split('.')[0] + '.' + str(i + 1)
            new_dict_list.append(new_dict)
        return new_dict_list

    def train(self, mode=True):
        from mmcv.utils.parrots_wrapper import _BatchNorm
        from vitmvt.utils import get_root_logger
        super(VIT_MVT_BACKBONE, self).train(mode)
        logger = get_root_logger()
        if self.use_inc_mode:
            for name, param in self.named_parameters():
                if 'ratio_specific' in name:
                    param.requires_grad = False


class TotalLayerSlice(SequentialSlice):
    """Generate layers due to depth ."""

    def __init__(self, stage_depth, embed_dims, norm_cfg, qkv_bias,
                 window_size, stage_settings, use_window_att, is_base):
        features = list()
        for i, layer_settings in enumerate(stage_settings):
            (mlp_ratio, num_heads, _, _) = layer_settings
            features.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    window_size=window_size,
                    norm_cfg=norm_cfg,
                    qkv_bias=qkv_bias,
                    use_window_att=use_window_att,
                    layer_indices=i,
                    is_base=is_base))

        super(TotalLayerSlice, self).__init__(*features, depth=stage_depth)


class TransformerEncoderLayer(BaseBackbone):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 mlp_ratio,
                 window_size=(14, 14),
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 layer_indices=0,
                 use_window_att=False,
                 is_base=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LayerNormSlice', eps=1e-6),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.attn = AttentionSlice(
            super_embed_dim=embed_dims,
            num_heads=self.num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            qkv_bias=qkv_bias,
            window_size=window_size,
            use_window_att=use_window_att)

        self.se = Channel_interaction(embed_dims, ratio=8, conv_cfg=dict(type='Conv2dSlice'))

        self.norm2_name, norm2 = build_norm_layer(
        norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.fc1 = LinearRatioSlice(embed_dims, mlp_ratio, ration_type='in')
        self.fc2 = LinearRatioSlice(embed_dims, mlp_ratio, ration_type='out')
        self.activate = build_activation_layer(act_cfg)
       
        from vitmvt.models import make_divisible
        hidden_dim = make_divisible(embed_dims * mlp_ratio, 1)
        self.dw = DWConv(hidden_dim)
        self.dw_init = DWConv(embed_dims)

        if is_base:
            if layer_indices == 3 or layer_indices == 7 or layer_indices == 11 or layer_indices == 15:
                if use_window_att:
                    self.aggregation = True
                else:
                    self.aggregation = False
            else:
                self.aggregation = False
        else:
            
            if layer_indices == 2 or layer_indices == 5 or layer_indices == 9 or layer_indices == 13:    
                if use_window_att:
                    self.aggregation = True
                else:
                    self.aggregation = False
            else:
                self.aggregation = False

        if self.aggregation:
            self.cnn_aggregation = BasicBlock(
                inplanes=self.embed_dims,
                planes=self.embed_dims,
                conv_cfg=dict(type='Conv2dSlice'),
                norm_cfg=dict(type='SyncBatchNormSlice'))

        from mmcv.cnn.bricks.transformer import build_dropout
        dropout_layer = dict(type='DropPath', drop_prob=0.1)
        self.dropout_layer = build_dropout(dropout_layer)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x, Hp, Wp):
        
        # lpu
        x = self.dw_init(x, Hp, Wp)

        residual = x
        x = self.norm1(x)

        B, _, C = x.shape
        x = self.se(self.attn(x, Hp, Wp).reshape(B, Hp, Wp, C).permute(
            0, 3, 1, 2).contiguous()).reshape(B, -1, Hp * Wp).permute(0, 2, 1).contiguous()

        x = residual + x

        B, _, C = x.shape
        
        residual = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.dw(x, Hp, Wp)
        x = self.activate(x)
        x = self.fc2(x)
        x = self.dropout_layer(x)
        x = residual + x

        if self.aggregation:
            x = self.cnn_aggregation(x, Hp, Wp)
        return x

def generate_supernet_settings(depth, mlp_ratio, num_heads):
    """Supernet settings case: Modify this method, if you want to search other
    fixed settings.

    Args:
        mlp_ratio:  the ratio of hidden dimension to the embedding
                    dimension in the multi-layer perceptron
        num_heads: number of heads
    Returns:
        list of list: The generated supernet setting for all the layers.
    """

    supernet_settings = []
    stage_num = depth('max')
    act_func = ['MemoryEfficientSwish'] * stage_num

    in_channel = 3
    for i in range(stage_num):
        mlp_ratio_single = mlp_ratio[i]
        num_heads_single = num_heads[i]
        act = act_func[i]
        layer_setting = [mlp_ratio_single, num_heads_single, act, in_channel]
        supernet_settings.append(layer_setting)
    return supernet_settings
