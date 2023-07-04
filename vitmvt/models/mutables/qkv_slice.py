import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MUTABLES
from .base_mutable import SliceOp


class QKV_Super(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 uniform_=None,
                 non_linear='linear',
                 scale=False):
        super().__init__(in_features, out_features, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        sample_weight = self.weight[:, :self.in_features]
        sample_weight = torch.cat(
            [sample_weight[i:self.out_features:3, :] for i in range(3)], dim=0)
        return F.linear(x, sample_weight, self.bias[:self.out_features])


class QKV_Super_score(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 uniform_=None,
                 non_linear='linear',
                 scale=False):
        super().__init__(in_features, out_features, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter('specific_weight', nn.Parameter(self.weight.clone()))
        self.register_parameter('specific_bias', nn.Parameter(self.bias.clone()))
        self.score = nn.Parameter(torch.rand(1))

    def forward(self, x):
        score = binarizer_fn(self.score)
        share_weight = self.weight[:, :self.in_features]
        share_weight = torch.cat(
            [share_weight[i:self.out_features:3, :] for i in range(3)], dim=0)
        
        spe_weight = self.weight[:, :self.in_features]
        spe_weight = torch.cat(
            [spe_weight[i:self.out_features:3, :] for i in range(3)], dim=0)

        sample_weight = score * share_weight + (1-score) * spe_weight
        if self.bias is not None:
            sample_bias = score * self.bias + (1-score) * self.specific_bias
        else:
            sample_bias = None

        return F.linear(x, sample_weight, sample_bias)


@MUTABLES.register_module()
class QkvSlice(nn.Linear, SliceOp):
    """Sliceable Linear module.

    Args:
        in_features (int/Int): The same with Linear.
        num_heads (int/Int): Parallel attention heads.
        bias (bool): The same with Linear. Defaults to True.
    """

    def __init__(self, in_features, num_heads, unit=64, bias=True, key=None):
        # Must initionalize first for multiple inheritance.
        SliceOp.__init__(self, key=key)
        max_in_features = self.get_value(in_features, kind='max')
        max_heads = self.get_value(num_heads, kind='max')
        max_out_features = max_heads * unit * 3
        super(QkvSlice, self).__init__(max_in_features, max_out_features, bias)
        self.in_features = in_features
        self.out_features = num_heads * unit * 3
        self.num_heads = num_heads
        self.unit = unit

    def forward_inner(self, x):
        in_features = self.get_value(self.in_features)
        out_features = self.get_value(self.num_heads) * self.unit * 3
        sample_weight = self.weight[:, :in_features]
        sample_weight = torch.cat(
            [sample_weight[i:out_features:3, :] for i in range(3)], dim=0)
        bias = self.bias[:out_features]
        return F.linear(x, sample_weight, bias)

    def forward(self, *args, **kwargs):
        """For multiple inheritance, we use the SliceOp's ``forward`` method to
        proxy.

        Note that, each mutiple inheritance mutable should realize this.
        """
        return SliceOp.forward(self, *args, **kwargs)

    def export(self, **kwargs):
        """Export LinearSlice to nn.Linear."""
        in_features = kwargs.get('in_features',
                                 self.get_value(self.in_features))
        num_heads = kwargs.get('num_heads', self.get_value(self.num_heads))
        out_features = self.unit * num_heads * 3
        sample_weight = self.weight[:out_features, :in_features]
        sample_weight = sample_weight.data
        export_module = QKV_Super(
            in_features, out_features, bias=(self.bias is not None))
        export_module.weight.data.copy_(sample_weight)
        if self.bias is not None:
            export_module.bias.data.copy_(self.bias.data[:out_features])
        return export_module


@MUTABLES.register_module()
class QkvSlice_ratio(nn.Linear, SliceOp):
    """Sliceable Linear module.

    Args:
        in_features (int/Int): The same with Linear.
        num_heads (int/Int): Parallel attention heads.
        bias (bool): The same with Linear. Defaults to True.
    """

    def __init__(self, in_features, num_heads, unit=64, bias=True, key=None):
        # Must initionalize first for multiple inheritance.
        SliceOp.__init__(self, key=key)
        max_in_features = self.get_value(in_features, kind='max')
        max_heads = self.get_value(num_heads, kind='max')
        max_out_features = max_heads * unit * 3
        super(QkvSlice_ratio, self).__init__(max_in_features, max_out_features, bias)
        self.in_features = in_features
        self.out_features = num_heads * unit * 3
        self.num_heads = num_heads
        self.unit = unit

        self.register_parameter('specific_weight', nn.Parameter(self.weight.clone()))
        self.register_parameter('specific_bias', nn.Parameter(self.bias.clone()))

        self.score = nn.Parameter(torch.rand(1))

    def forward_inner(self, x):
        # self.num_heads.set_curr(num_heads)
        in_features = self.get_value(self.in_features)
        out_features = self.get_value(self.num_heads) * self.unit * 3

        sha_weight = self.weight[:out_features, :in_features]
        spe_weight = self.specific_weight[:out_features, :in_features]

        if self.bias is not None:
            sha_bias = self.bias[:out_features]
            spe_bias = self.specific_bias[:out_features]

        score = binarizer_fn(self.score)

        weight = score * sha_weight + (1-score) * spe_weight
        weight = torch.cat(
            [weight[i:out_features:3, :] for i in range(3)], dim=0)

        if self.bias is not None:
            bias = score * sha_bias + (1-score) * spe_bias
        else:
            bias = None

        return F.linear(x, weight, sample_bias)

    def forward(self, *args, **kwargs):
        """For multiple inheritance, we use the SliceOp's ``forward`` method to
        proxy.

        Note that, each mutiple inheritance mutable should realize this.
        """
        return SliceOp.forward(self, *args, **kwargs)

    def export(self, **kwargs):
        """Export LinearSlice to nn.Linear."""
        in_features = kwargs.get('in_features',
                                 self.get_value(self.in_features))
        num_heads = kwargs.get('num_heads', self.get_value(self.num_heads))
        
        out_features = self.unit * num_heads * 3

        sha_weight = self.weight[:out_features, :in_features]
        spe_weight = self.specific_weight[:out_features, :in_features]

        if self.bias is not None:
            sha_bias = self.bias[:out_features]
            spe_bias = self.specific_bias[:out_features]

        score = binarizer_fn(self.score)
        weight = score * sha_weight + (1-score) * spe_weight
        if self.bias is not None:
            bias = score * sha_bias + (1-score) * spe_bias
        else:
            bias = None

        export_module = QKV_Super_score(
            in_features, out_features, bias=(self.bias is not None))
        export_module.weight.data.copy_(weight.data)
        if self.bias is not None:
            export_module.bias.data.copy_(bias.data)
        return export_module


class BinarizerFn(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold=0.5):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput, None

binarizer_fn = BinarizerFn.apply

