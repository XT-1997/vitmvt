import copy

import torch.nn as nn

from ..builder import MUTABLES
from .base_mutable import SliceOp


@MUTABLES.register_module()
class SequentialSlice(nn.Sequential, SliceOp):
    """Sliceable Sequential module.

    Args:
        args (list): Same as Sequential.
        depth (Space/None, optional): Depth space. Defaults to None.
    """

    def __init__(self, *args, depth=None, key=None):
        SliceOp.__init__(self, key=key)
        super(SequentialSlice, self).__init__(*args)
        if depth is None:
            depth = len(self)
        self.depth = depth

    def forward(self, *args, **kwargs):
        """For multiple inheritance, we use the SliceOp's ``forward`` method to
        proxy.

        Note that, each mutiple inheritance mutable should realize this.
        """
        return SliceOp.forward(self, *args, **kwargs)

    def forward_inner(self, input):
        num_blocks = self.get_value(self.depth)
        assert num_blocks <= len(self), 'num blocks should <= len(self), \
            {} vs {}.'.format(num_blocks, len(self))
        for idx, submodule in enumerate(self._modules.values()):
            if idx >= num_blocks:
                break
            input = submodule(input)
        return input

    def extra_repr(self):
        depth = copy.deepcopy(self.depth)
        if self.get_value(
                depth, kind='max') != self.get_value(
                    depth, kind='min'):
            return f'depth={self.depth}'

    def export(self, **kwargs):
        export_module = nn.Sequential()
        num_blocks = kwargs.get('depth', self.get_value(self.depth))
        idx = 0
        for name in self._modules:
            if idx >= num_blocks:
                break
            else:
                layer = self.__getattr__(name)
                if isinstance(layer, SliceOp):
                    export_module.add_module(name, layer.export())
                else:
                    export_module.add_module(name, layer)
            idx += 1
        return export_module
