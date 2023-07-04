# yapf:disable
from .algorithms import ViT_MVT
from .architectures import (EncoderDecoderSearch, ImageClassifierSearch,
                            MaskRCNNSearch)
from .backbones import VIT_MVT_BACKBONE
from .builder import (ALGORITHMS, ARCHITECTURES, BACKBONES, HEADS, LOSSES,
                      MUTABLES, MUTATORS, NECKS, OPS, build_algorithm,
                      build_backbone, build_distiller, build_head, build_loss,
                      build_model, build_mutable, build_mutator, build_neck,
                      build_op)
from .heads import LinearClsHead
from .mutables import (AttentionSlice, BaseMutable, BatchNorm2dSlice,
                       BatchNorm3dSlice, Conv1dSlice, Conv2dSlice,
                       Conv3dSlice, ConvModuleSlice, ConvTranspose2dSlice,
                       LayerNormSlice, LayerNormSlice_ratio, LinearSlice, 
                       LinearSlice_byhead, MixedOp, DynamicLinearSlice,
                       ParameterSlice, PatchembedSlice, PReLUSlice, QkvSlice,
                       QkvSlice_ratio, RelativePositionSlice2D, Resize3DSlice,
                       ResizeSlice, SequentialSlice, SliceOp,
                       SyncBatchNormSlice)
from .mutators import RandomMutator, StateslessMutator
from .necks import SFP
from .ops import Resize, Resize3D
from .losses import Accuracy
from .utils import (AlphaLayer, compactor_convert, dmcp_make_divisible,
                    drop_path, make_divisible)

# yapf: disable
__all__ = [
    # ALGORITHMS
    'ViT_MVT'
    # ARCHTECTURES
    'ImageClassifierSearch', 'MaskRCNNSearch', 'EncoderDecoderSearch',
    # BACKBONES
    'VIT_MVT_BACKBONE',
    # BUILDERS
    'ALGORITHMS', 'ARCHITECTURES', 'BACKBONES', 'HEADS', 'LOSSES',
    'MUTABLES', 'MUTATORS', 'NECKS', 'OPS', 'build_algorithm',
    'build_backbone', 'build_distiller', 'build_head', 'build_loss',
    'build_model', 'build_mutable', 'build_mutator', 'build_neck',
    'build_op',
    # HEADS
    'LinearClsHead',
    # MUTABLES
    'BaseMutable', 'BatchNorm2dSlice', 'BatchNorm3dSlice',
    'Conv1dSlice', 'Conv2dSlice', 'Conv3dSlice', 'ConvModuleSlice',
    'ConvTranspose2dSlice', 'LinearSlice',
    'MixedOp', 'PReLUSlice', 'Resize3DSlice', 'ResizeSlice',
    'SequentialSlice', 'SliceOp', 'SyncBatchNormSlice',
    'LayerNormSlice', 'RelativePositionSlice2D',
    'AttentionSlice', 'LinearSlice_byhead', 'ParameterSlice',
    'PatchembedSlice', 'QkvSlice',  
    'LayerNormSlice_ratio', 
    'QkvSlice_ratio', 'DynamicLinearSlice',
    # MUTATORS
    'DMCPMutator', 'RandomMutator', 'StateslessMutator', 'DartsMutator',
    'ListMutator', 'Cream_Mutator', 'ResRepMutator', 'DCFFMutator',
    # NECKS
    'SFP',
    # OPS
    'Resize', 'Resize3D',
    # 'LOSSES
    'Accuracy',
    # UTILS
    'make_divisible', 'drop_path', 'compactor_convert', 'dmcp_make_divisible',
    'AlphaLayer'
    # ARCHITECTURES
]
# yapf: enable
