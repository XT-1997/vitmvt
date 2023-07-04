from .act_slice import PReLUSlice
from .attention_slice import AttentionSlice
from .base_mutable import BaseMutable, MixedOp, SliceOp
from .bn_slice import BatchNorm2dSlice, BatchNorm3dSlice, SyncBatchNormSlice
from .container_slice import SequentialSlice
from .conv_module_slice import ConvModuleSlice
from .conv_slice import (Conv1dSlice, Conv2dSlice, Conv3dSlice,
                         ConvTranspose2dSlice)
from .interp_slice import Resize3DSlice, ResizeSlice
from .linear_slice import LinearRatioSlice, LinearSlice, LinearSlice_byhead, DynamicLinearSlice
from .ln_slice import LayerNormSlice, LayerNormSlice_ratio
from .parameter_slice import ParameterSlice
from .patchembed_slice import PatchembedSlice
from .qkv_slice import QkvSlice, QkvSlice_ratio
from .relative_position_slice import RelativePositionSlice2D

# yapf: disable
__all__ = [
    'PReLUSlice', 'BaseMutable', 'MixedOp', 'SliceOp', 'BatchNorm2dSlice',
    'BatchNorm3dSlice', 'SyncBatchNormSlice', 'Conv1dSlice', 'Conv2dSlice',
    'Conv3dSlice', 'ConvTranspose2dSlice', 'ResizeSlice', 'Resize3DSlice',
    'ConvModuleSlice', 'DynamicLinearSlice', 'LinearSlice', 'SequentialSlice',
    'LayerNormSlice', 'RelativePositionSlice2D',
    'LinearRatioSlice', 'AttentionSlice', 
    'ParameterSlice', 'PatchembedSlice', 'QkvSlice', 'DMCPConvModuleSlice',  
    'QkvSlice_ratio', 'LayerNormSlice_ratio',
    'LinearSlice_byhead', 'DynamicLinearSlice'
]
# yapf: enable
