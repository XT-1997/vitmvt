import numpy as np
import torch.nn.functional as F

from vitmvt.space import build_search_space
from ..builder import MUTABLES
from ..ops import Resize, Resize3D
from .base_mutable import SliceOp


@MUTABLES.register_module()
class ResizeSlice(SliceOp):
    """Sliceable Resize module.

    Args:
        resolution (int or Int space): output spatial resolution
        mode (string): interp algo
    """

    def __init__(self, resolution, mode='bilinear', key=None):
        super(ResizeSlice, self).__init__(key=key)
        if isinstance(resolution, dict):
            resolution = build_search_space(resolution)
        self.resolution = resolution
        self.mode = mode

    def forward_inner(self, x):
        _, _, H, W = x.size()
        h, w = self.ntuple(self.get_value(self.resolution))
        if (H, W) != (h, w):
            x = F.interpolate(
                x,
                size=self.ntuple(self.get_value(self.resolution)),
                mode=self.mode)
        return x

    def ntuple(self, attr):
        if isinstance(attr, (tuple, list)):
            return attr
        return (attr, attr)

    def extra_repr(self):
        s = []
        s.append('{resolution}')
        s.append('mode={mode}')
        return ', '.join(s).format(**self.__dict__)

    def export(self, **kwargs):
        resolution = kwargs.get('resolution', self.get_value(self.resolution))
        normal_resize = Resize(resolution, mode=self.mode)
        return normal_resize


@MUTABLES.register_module()
class Resize3DSlice(SliceOp):
    """Sliceable Resize3D module.

    Args:
        resolution (None, int or Int space): output spatial resolution
        length (None, int or Int space): output temporal length
        mode (string): interp algo. Defaults to `bilinear`.
    """

    def __init__(self, resolution, length, mode='bilinear', key=None):
        super(Resize3DSlice, self).__init__(key=key)
        if isinstance(resolution, dict):
            resolution = build_search_space(resolution)
        if isinstance(length, dict):
            length = build_search_space(length)
        self.resolution = resolution
        self.length = length
        self.mode = mode

    def forward_inner(self, x):
        *others, T, H, W = x.size()

        if self.length is not None:
            t = self.get_value(self.length)
            assert t <= T, f'required {t} exceed max length {T}'
            avg_interval = T // t
            if avg_interval > 0:
                base_offsets = np.arange(t) * avg_interval
                if self.training:
                    snips_offnets = np.random.randint(avg_interval, size=t)
                else:
                    snips_offnets = avg_interval // 2
                frame_inds = base_offsets + snips_offnets
            else:
                # avg_interval == 0:
                frame_inds = np.around(np.arange(t) * (T / t))
            xt = x[:, :, frame_inds, ...]
        else:
            t = T
            xt = x

        if self.resolution is not None:
            h, w = self.ntuple(self.get_value(self.resolution))
            if (H, W) != (h, w):
                xt = xt.reshape(-1, t, H, W)
                xt = F.interpolate(
                    xt,
                    size=self.ntuple(self.get_value(self.resolution)),
                    mode=self.mode)
                xt = xt.reshape(tuple(others) + (t, h, w))
        return xt

    def ntuple(self, attr):
        if isinstance(attr, (tuple, list)):
            return attr
        return (attr, attr)

    def extra_repr(self):
        s = []
        s.append('{resolution}')
        s.append('{length}')
        s.append('mode={mode}')
        return ', '.join(s).format(**self.__dict__)

    def export(self, **kwargs):
        resolution = kwargs.get('resolution', self.get_value(self.resolution))
        length = kwargs.get('length', self.get_value(self.length))
        resize3dSlice = Resize3D(resolution, length, self.mode)
        return resize3dSlice
