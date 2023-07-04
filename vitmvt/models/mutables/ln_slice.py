import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import NORM_LAYERS

from ..builder import MUTATORS
from .base_mutable import SliceOp

DEFAULT_SHARED_MODE = 'mask'


@MUTATORS.register_module()
@NORM_LAYERS.register_module()
class LayerNormSlice(nn.LayerNorm, SliceOp):
    """Sliceable LayerNorm module.

    Args:
        embedding_dim (int/Int): The same with LayerNorm.
    """

    def __init__(self, embedding_dim, eps=1e-5, key=None):
        assert isinstance(embedding_dim.curr,
                          int), 'Only support Transformer Example'
        # Must initionalize first for multiple inheritance.
        SliceOp.__init__(self, key=key)
        max_embedding_dim = self.get_value(embedding_dim, kind='max')
        super(LayerNormSlice, self).__init__(max_embedding_dim, eps)
        self.embedding_dim = embedding_dim

    def forward_inner(self, x):
        embedding_dim = self.get_value(self.embedding_dim)
        weight = self.weight[:embedding_dim]
        if self.bias is None:
            bias = None
        else:
            bias = self.bias[:embedding_dim]

        return F.layer_norm(x, (embedding_dim, ), weight, bias, self.eps)

    def forward(self, *args, **kwargs):
        """For multiple inheritance, we use the SliceOp's ``forward`` method to
        proxy.

        Note that, each mutiple inheritance mutable should realize this.
        """
        return SliceOp.forward(self, *args, **kwargs)

    def export(self, **kwargs):
        """Export LayerNormSlice to nn.LayerNorm."""
        embedding_dim = kwargs.get('embedding_dim',
                                   self.get_value(self.embedding_dim))

        export_module = nn.LayerNorm(embedding_dim)
        export_module.weight.data.copy_(self.weight.data[:embedding_dim])
        if self.bias is not None:
            export_module.bias.data.copy_(self.bias.data[:embedding_dim])

        return export_module


@MUTATORS.register_module()
@NORM_LAYERS.register_module()
class LayerNormSlice_ratio(nn.LayerNorm, SliceOp):
    """Sliceable LayerNorm module.

    Args:
        embedding_dim (int/Int): The same with LayerNorm.
    """

    def __init__(self, embedding_dim, share_ratio=None, shared_mode=DEFAULT_SHARED_MODE, eps=1e-5, key=None):
        assert isinstance(embedding_dim.curr,
                          int), 'Only support Transformer Example'
        # Must initionalize first for multiple inheritance.
        SliceOp.__init__(self, key=key)
        max_embedding_dim = self.get_value(embedding_dim, kind='max')
        super(LayerNormSlice_ratio, self).__init__(max_embedding_dim, eps)
        self.embedding_dim = embedding_dim
        self.share_ratio = share_ratio
        self.register_parameter('specific_weight', nn.Parameter(self.weight.clone()))
        self.register_parameter('specific_bias', nn.Parameter(self.bias.clone()))
        
        self.shared_mode = shared_mode

        self.ratio_specific = nn.Parameter(torch.rand(max_embedding_dim))

    def slice_params(self, embedding_dim, ratio):
        
        if isinstance(ratio, float):
            share_weight = self.weight[:int(embedding_dim * ratio)]
            spec_weight = self.specific_weight[int(embedding_dim*ratio):embedding_dim]
            weight = torch.cat([share_weight, spec_weight], dim=0).contiguous()
        else:
            share_weight = self.weight[:embedding_dim]
            spec_weight = self.specific_weight[:embedding_dim]
            p_ratio = ratio[:embedding_dim]
            weight = p_ratio * share_weight + (1-p_ratio) * spec_weight

        if self.bias is None:
            bias = None
        else:
            if isinstance(ratio, float):
                shared_bias = self.bias[:int(embedding_dim * ratio)]
                specific_bias = self.specific_weight[int(embedding_dim * ratio):embedding_dim]
                bias = torch.cat([shared_bias, specific_bias], dim=0) 
            else:
                share_bias = self.bias[:embedding_dim]
                spec_bias = self.specific_bias[:embedding_dim]
                p_ratio = ratio[:embedding_dim]
                # import mmcv
                # rank, world_size = mmcv.runner.get_dist_info()
                # if rank == 0:
                #     with open("/mnt/cache/xietao/ln.txt", "a") as f:
                #         f.write(str(p_ratio.nonzero().shape[0] / p_ratio.shape[0]))
                #         f.write('\n')
                bias = p_ratio * share_bias + (1-p_ratio) * spec_bias

        return weight, bias

    def forward_inner(self, x):
        embedding_dim = self.get_value(self.embedding_dim)
        # ratio = self.get_value(self.share_ratio)
        if self.shared_mode == 'mask':
            ratio = binarizer_fn(self.ratio_specific, self.get_value(self.share_ratio))
        else:
            ratio = self.get_value(self.share_ratio)

        weight, bias = self.slice_params(embedding_dim, ratio)
        return F.layer_norm(x, (embedding_dim, ), weight, bias, self.eps)

    def forward(self, *args, **kwargs):
        """For multiple inheritance, we use the SliceOp's ``forward`` method to
        proxy.

        Note that, each mutiple inheritance mutable should realize this.
        """
        return SliceOp.forward(self, *args, **kwargs)

    def export(self, **kwargs):
        """Export LayerNormSlice to nn.LayerNorm."""
        embedding_dim = kwargs.get('embedding_dim',
                                   self.get_value(self.embedding_dim))
        # ratio = kwargs.get('share_ratio',
        #                            self.get_value(self.share_ratio))

        if self.shared_mode == 'mask':
            ratio = binarizer_fn(self.ratio_specific, self.get_value(self.share_ratio))
        else:
            ratio = self.get_value(self.share_ratio)

        export_module = nn.LayerNorm(embedding_dim)
        weight, bias = self.slice_params(embedding_dim, ratio)

        export_module.weight.data.copy_(weight.data)
        if self.bias is not None:
            export_module.bias.data.copy_(bias.data)

        return export_module


class BinarizerFn(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput, None

binarizer_fn = BinarizerFn.apply

