import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..builder import OPS


@OPS.register_module()
class Resize(nn.Module):

    def __init__(self, resolution, mode='bilinear'):
        super(Resize, self).__init__()
        self.resolution = resolution
        self.mode = mode

    def forward(self, x):
        _, _, H, W = x.size()
        h, w = self.ntuple(self.resolution)
        if (H, W) != (h, w):
            x = F.interpolate(
                x, size=self.ntuple(self.resolution), mode=self.mode)
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


@OPS.register_module()
class Resize3D(nn.Module):

    def __init__(self, resolution, length, mode='bilinear'):
        super(Resize3D, self).__init__()
        self.resolution = resolution
        self.length = length
        self.mode = mode

    def forward(self, x):
        *others, T, H, W = x.size()

        if self.length is not None:
            t = self.length
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
            h, w = self.ntuple(self.resolution)
            if (H, W) != (h, w):
                xt = xt.reshape(-1, t, H, W)
                xt = F.interpolate(
                    xt, size=self.ntuple(self.resolution), mode=self.mode)
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
