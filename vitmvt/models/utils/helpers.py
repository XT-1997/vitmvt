# Copyright (c) OpenMMLab. All rights reserved.
import collections.abc
import warnings
from itertools import repeat

import torch
from mmcv.utils import digit_version

from vitmvt.space import FreeBindAPI


def make_divisible(v, divisor=8, min_value=None):

    def make_divisible_wrapper(v, divisor=8, min_value=None):
        min_value = min_value or divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    if isinstance(v, FreeBindAPI):
        return v._new_obj_binary(
            v, divisor, make_divisible_wrapper, min_value=min_value)
    else:
        return make_divisible_wrapper(v, divisor=divisor, min_value=min_value)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet,
    etc networks, however, the original name is misleading as 'Drop Connect'
    is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    I've opted for changing the layer and argument names to 'drop path'
    rather than mix DropConnect as a layer name and use 'survival rate'
    as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def is_tracing() -> bool:
    if digit_version(torch.__version__) >= digit_version('1.6.0'):
        on_trace = torch.jit.is_tracing()
        # In PyTorch 1.6, torch.jit.is_tracing has a bug.
        # Refers to https://github.com/pytorch/pytorch/issues/42448
        if isinstance(on_trace, bool):
            return on_trace
        else:
            return torch._C._is_tracing()
    else:
        warnings.warn(
            'torch.jit.is_tracing is only supported after v1.6.0. '
            'Therefore is_tracing returns False automatically. Please '
            'set on_trace manually if you are using trace.', UserWarning)
        return False


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
