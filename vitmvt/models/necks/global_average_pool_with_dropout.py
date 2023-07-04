# Modified from mmcls.models.GlobalAveragePooling
# (https://gitlab.sz.sensetime.com/parrotsDL-sz/mmclassification/-/blob/master/mmcls/models/necks/gap.py)
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS


@NECKS.register_module()
class GlobalAveragePoolingWithDropout(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dropout=None, dim=2):
        super(GlobalAveragePoolingWithDropout, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = dropout

    def init_weights(self):
        pass

    def forward(self, inputs, drop_ratio=None):
        drop_ratio = drop_ratio if drop_ratio is not None else self.dropout
        drop_ratio = 0.0 if drop_ratio is None else drop_ratio
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            if drop_ratio > 0 and self.training:
                outs = tuple([F.dropout(x, p=drop_ratio) for x in outs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            if drop_ratio > 0 and self.training:
                outs = F.dropout(outs, p=drop_ratio)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
