import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ACTIVATION_LAYERS
from torch.nn import Parameter

from ..builder import MUTATORS
from .base_mutable import SliceOp


@MUTATORS.register_module()
@ACTIVATION_LAYERS.register_module()
class PReLUSlice(SliceOp):
    """Sliceable PReLU module.

    Args:
        num_parameters (int/Int, optional): Same as PReLU. Defaults to 1.
        init (float, optional): Same as PReLU. Defaults to 0.25.
    """

    def __init__(self, num_parameters=1, init=0.25, key=None):
        super(PReLUSlice, self).__init__(key=key)
        self.num_parameters = num_parameters
        self.weight = Parameter(
            torch.Tensor(self.get_value(num_parameters,
                                        kind='max')).fill_(init))

    def forward_inner(self, input):
        num_parameters = self.get_value(self.num_parameters)
        weight = self.weight[:num_parameters]
        return F.prelu(input, weight)

    def extra_repr(self):
        return 'num_parameters={}'.format(self.num_parameters)

    def export(self, **kwargs):
        """Export PReLUSlice to nn.PReLU."""
        num_parameters = kwargs.get('num_parameters',
                                    self.get_value(self.num_parameters))

        export_module = nn.PReLU(num_parameters=num_parameters)
        weight = self.weight[:num_parameters]
        export_module.weight.data.copy_(weight)
        return export_module
