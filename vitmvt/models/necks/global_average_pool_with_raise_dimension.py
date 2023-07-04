# Modified from mmcls.models.GlobalAveragePooling
# (https://gitlab.sz.sensetime.com/parrotsDL-sz/mmclassification/-/blob/master/mmcls/models/necks/gap.py)
import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class GlobalAveragePoolingWithConv(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self,
                 in_channels=320,
                 num_features=1280,
                 dim=2,
                 dropout=None):
        super(GlobalAveragePoolingWithConv, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv_head = nn.Conv2d(in_channels, num_features, 1)
        self.dropout = nn.Dropout(dropout) if dropout is not None else dropout

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple([self.conv_head(x) for x in outs])
            outs = tuple([x.mul_(x.sigmoid()) for x in outs])
            if self.dropout is not None:
                outs = tuple([self.dropout(x) for x in outs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = self.conv_head(outs)
            outs = outs.mul_(outs.sigmoid())
            if self.dropout is not None:
                outs = self.dropout(outs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
