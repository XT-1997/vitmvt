import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MUTABLES
from .base_mutable import SliceOp


# used for supernet pretrain
@MUTABLES.register_module()
class LinearSlice(nn.Linear, SliceOp):
    """Sliceable Linear module.

    Args:
        in_features (int/Int): The same with Linear.
        out_features (int/Int): The same with Linear.
        bias (bool): The same with Linear. Defaults to True.
    """

    def __init__(self, in_features, out_features, bias=True, key=None):
        # Must initionalize first for multiple inheritance.
        SliceOp.__init__(self, key=key)
        max_in_features = self.get_value(in_features, kind='max')
        max_out_features = self.get_value(out_features, kind='max')
        super(LinearSlice, self).__init__(max_in_features, max_out_features,
                                          bias)
        self.in_features = in_features
        self.out_features = out_features

    def forward_inner(self, x):
        in_features = self.get_value(self.in_features)
        out_features = self.get_value(self.out_features)
        weight = self.weight[:out_features, :in_features].contiguous()
        if self.bias is None:
            bias = None
        else:
            bias = self.bias[:out_features]
        return F.linear(x, weight, bias)

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
        out_features = kwargs.get('out_features',
                                  self.get_value(self.out_features))
        export_module = Linear_mvt(
            in_features, out_features, bias=(self.bias is not None))
        export_module.weight.data.copy_(
            self.weight.data[:out_features, :in_features])
        if self.bias is not None:
            export_module.bias.data.copy_(self.bias.data[:out_features])

        return export_module


# used for subnet test
class Linear_mvt(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)

        self.register_parameter('specific_weight', nn.Parameter(self.weight.clone()))
        if self.bias is not None:
            self.register_parameter('specific_bias', nn.Parameter(self.bias.clone()))
        self.score = nn.Parameter(torch.rand(1))

    def forward(self, input):
        score = binarizer_fn(self.score)
        weight = score * self.weight + (1-score) * self.specific_weight
        if self.bias is not None:
            bias = score * self.bias + (1-score) * self.specific_bias
        else:
            bias = None
        return F.linear(input, weight, bias)


# used for supernet finetune for all tasks
@MUTABLES.register_module()
class LinearRatioSlice(nn.Linear, SliceOp):
    """Sliceable Linear module.

    Args:
        in_features (int/Int): The same with Linear.
        mlp_ration (int/Int): the ratio of hidden dimension
                              to the embedding dimension in
                              the multi-layer perceptron)
        bias (bool): The same with Linear. Defaults to True.
    """

    def __init__(self,
                 in_features,
                 mlp_ration,
                 ration_type='in',
                 bias=True,
                 key=None):
        # Must initionalize first for multiple inheritance.
        SliceOp.__init__(self, key=key)
        max_in_features = self.get_value(in_features, kind='max')
        max_ratio = self.get_value(mlp_ration, kind='max')
        max_in_out_features = max_in_features * max_ratio
        self.ration_type = ration_type
        if self.ration_type == 'in':
            super(LinearRatioSlice, self).__init__(max_in_features,
                                                   max_in_out_features, bias)
        else:
            super(LinearRatioSlice, self).__init__(max_in_out_features,
                                                   max_in_features, bias)
        self.in_features = in_features
        self.out_features = in_features * mlp_ration
        self.ratio = mlp_ration

        self.register_parameter('specific_weight', nn.Parameter(self.weight.clone()))
        self.register_parameter('specific_bias', nn.Parameter(self.bias.clone()))
        self.score = nn.Parameter(torch.rand(1))

    def forward_inner(self, x):
        if self.ration_type == 'in':
            in_features = self.get_value(self.in_features)
            ratio = self.get_value(self.ratio)
            out_features = int(in_features * ratio)
            sha_weight = self.weight[:out_features, :in_features].contiguous()
            spe_weight = self.specific_weight[:out_features, :in_features].contiguous()
        else:
            out_features = self.get_value(self.in_features)
            ratio = self.get_value(self.ratio)
            in_features = int(out_features * ratio)
            sha_weight = self.weight[:out_features, :in_features].contiguous()
            spe_weight = self.specific_weight[:out_features, :in_features].contiguous()

        if self.bias is None:
            sha_bias = None
            spe_bias = None
        else:
            sha_bias = self.bias[:out_features]
            spe_bias = self.specific_bias[:out_features]
        
        # adaptive
        score = binarizer_fn(self.score)
        weight = score * sha_weight + (1-score) * spe_weight
        if self.bias is not None:
            bias = score * sha_bias + (1-score) * spe_bias
        else:
            bias = None

        return F.linear(x, weight, bias)

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
        ratio = kwargs.get('mlp_ration', self.get_value(self.ratio))
        if self.ration_type == 'in':
            out_features = int(in_features * ratio)
            export_module = Linear_mvt(
                in_features, out_features, bias=(self.bias is not None))

            export_module.weight.data.copy_(
                self.weight.data[:out_features, :in_features])
            export_module.specific_weight.data.copy_(
                self.specific_weight.data[:out_features, :in_features])
            export_module.score.data.copy_(self.score.data)

            if self.bias is not None:
                export_module.bias.data.copy_(self.bias.data[:out_features])
                export_module.specific_bias.data.copy_(self.specific_bias.data[:out_features])
        else:
            out_features = int(in_features * ratio)
            export_module = Linear_mvt(
                out_features, in_features, bias=(self.bias is not None))
            export_module.weight.data.copy_(
                self.weight.data[:in_features, :out_features])
            export_module.specific_weight.data.copy_(
                self.specific_weight.data[:in_features, :out_features])
            export_module.score.data.copy_(self.score.data)

            if self.bias is not None:
                export_module.bias.data.copy_(self.bias.data[:in_features])
                export_module.specific_bias.data.copy_(self.specific_bias.data[:in_features])

        return export_module


@MUTABLES.register_module()
class DynamicLinearSlice(LinearSlice):
    """Sliceable Dynamic Linear module.

    Linear weight could be dynamic sliced based on the input shape.
    """

    def forward_inner(self, x):
        in_features = x.size(-1)
        assert in_features <= self.get_value(self.in_features, kind='max'), \
            'Required: data_size <= max weight_size, get ' \
            f"{in_features} vs {self.in_features('max')}"
        out_features = self.get_value(self.out_features)
        weight = self.weight[:out_features, :in_features]
        if self.bias is None:
            bias = None
        else:
            bias = self.bias[:out_features]
        return F.linear(x, weight, bias)


# used for supernet pretrain
@MUTABLES.register_module()
class LinearSlice_byhead(nn.Linear, SliceOp):
    """Sliceable Linear module.

    Args:
        out_features (int/Int): The same with Linear.
        num_heads (int/Int): Parallel heads.
        bias (bool): The same with Linear. Defaults to True.
    """

    def __init__(self, out_features, num_heads, unit=64, bias=True, key=None):
        # Must initionalize first for multiple inheritance.
        SliceOp.__init__(self, key=key)
        max_out_features = self.get_value(out_features, kind='max')
        max_heads = self.get_value(num_heads, kind='max')
        max_in_features = max_heads * unit
        super(LinearSlice_byhead, self).__init__(max_in_features,
                                                 max_out_features, bias)
        self.out_features = out_features
        self.in_features = unit * num_heads
        self.num_heads = num_heads
        self.unit = unit

    def forward_inner(self, x):
        out_features = self.get_value(self.out_features)
        in_features = self.get_value(self.num_heads) * self.unit
        weight = self.weight[:out_features, :in_features].contiguous()
        if self.bias is None:
            bias = None
        else:
            bias = self.bias[:out_features]
        return F.linear(x, weight, bias)

    def forward(self, *args, **kwargs):
        """For multiple inheritance, we use the SliceOp's ``forward`` method to
        proxy.

        Note that, each mutiple inheritance mutable should realize this.
        """
        return SliceOp.forward(self, *args, **kwargs)

    def export(self, **kwargs):
        """Export LinearSlice to nn.Linear."""
        out_features = kwargs.get('out_features',
                                  self.get_value(self.out_features))
        num_heads = kwargs.get('num_heads', self.get_value(self.num_heads))
        in_features = self.unit * num_heads
        sample_weight = self.weight[:out_features, :in_features]
        sample_weight = sample_weight.data
        export_module = Linear_mvt(
            in_features, out_features, bias=(self.bias is not None))
        export_module.weight.data.copy_(
            self.weight.data[:out_features, :in_features])
        if self.bias is not None:
            export_module.bias.data.copy_(self.bias.data[:out_features])
        return export_module


@MUTABLES.register_module()
class LinearSlice_byhead_ratio(nn.Linear, SliceOp):
    """Sliceable Linear module.

    Args:
        out_features (int/Int): The same with Linear.
        num_heads (int/Int): Parallel heads.
        bias (bool): The same with Linear. Defaults to True.
    """

    def __init__(self, out_features, num_heads, unit=64, bias=True, key=None):
        # Must initionalize first for multiple inheritance.
        SliceOp.__init__(self, key=key)
        max_out_features = self.get_value(out_features, kind='max')
        max_heads = self.get_value(num_heads, kind='max')
        max_in_features = max_heads * unit
        super(LinearSlice_byhead_ratio, self).__init__(max_in_features,
                                                       max_out_features, bias)
        self.out_features = out_features
        self.in_features = unit * num_heads
        self.num_heads = num_heads
        self.unit = unit
        self.register_parameter('specific_weight', nn.Parameter(self.weight.clone()))
        self.register_parameter('specific_bias', nn.Parameter(self.bias.clone()))
        self.score = nn.Parameter(torch.rand(1))

    def forward_inner(self, x):
        out_features = self.get_value(self.out_features)
        in_features = self.get_value(self.num_heads) * self.unit
        sha_weight = self.weight[:out_features, :in_features].contiguous()
        spe_weight = self.specific_weight[:out_features, :in_features].contiguous()

        if self.bias is None:
            sha_bias = None
            spe_bias = None
        else:
            sha_bias = self.bias[:out_features]
            spe_bias = self.specific_bias[:out_features]
        
        # adaptive
        score = binarizer_fn(self.score)
        weight = score * sha_weight + (1-score) * spe_weight
        if self.bias is not None:
            bias = score * sha_bias + (1-score) * spe_bias
        else:
            bias = None
        return F.linear(x, weight, bias)
    
    def forward(self, *args, **kwargs):
        """For multiple inheritance, we use the SliceOp's ``forward`` method to
        proxy.

        Note that, each mutiple inheritance mutable should realize this.
        """
        return SliceOp.forward(self, *args, **kwargs)

    def export(self, **kwargs):
        """Export LinearSlice to nn.Linear."""
        out_features = kwargs.get('out_features',
                                  self.get_value(self.out_features))
        num_heads = kwargs.get('num_heads', self.get_value(self.num_heads))
        in_features = self.unit * num_heads
        export_module = Linear_mvt(
            in_features, out_features, bias=(self.bias is not None))

        export_module.weight.data.copy_(
            self.weight.data[:out_features, :in_features])
        export_module.specific_weight.data.copy_(
            self.specific_weight.data[:out_features, :in_features])
        export_module.score.data.copy_(self.score.data)

        if self.bias is not None:
            export_module.bias.data.copy_(self.bias.data[:out_features])
            export_module.specific_bias.data.copy_(self.specific_bias.data[:out_features])

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
