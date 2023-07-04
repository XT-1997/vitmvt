import warnings

from mmcv.cnn import INITIALIZERS as MMCV_INITIALIZERS
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.utils import Registry

from vitmvt.utils import import_used_modules, replace_default_registry

MODELS = Registry('models', parent=MMCV_MODELS)

INITIALIZERS = Registry('initializer', parent=MMCV_INITIALIZERS)

# Temporary registry of mmcv
ROOT_MODEL_WRAPPERS = Registry('model_wrapper', scope='mmcv')
MODEL_WRAPPERS = Registry('model_wrapper', parent=ROOT_MODEL_WRAPPERS)

ROOT_MODEL_WRAPPERS.register_module(module=MMDataParallel)
ROOT_MODEL_WRAPPERS.register_module(module=MMDistributedDataParallel)

GENERATORS = MODELS
ALGORITHMS = MODELS
ARCHITECTURES = MODELS
BACKBONES = MODELS
HEADS = MODELS
LOSSES = MODELS
MUTABLES = MODELS
MUTATORS = MODELS
NECKS = MODELS
OPS = MODELS

# Unique distillation modules
CONNECTORS = MODELS
DISTILLERS = MODELS

# Unique quantizatioin modules
QUANTIZER = MODELS
OBSERVERS = MODELS
FAKE_QUANTIZATION = MODELS


def build_distiller(cfg):
    return DISTILLERS.build(cfg)


def build_connector(cfg):
    return CONNECTORS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_algorithm(cfg, registry=None):
    return ALGORITHMS.build(cfg)


def build_model(cfg):
    import_used_modules(cfg, 'models')

    from vitmvt.utils import DEFAULT_SCOPE
    if DEFAULT_SCOPE:
        registry = replace_default_registry(DEFAULT_SCOPE, 'MODELS',
                                            'models.builder')
        if registry:
            return registry.build(cfg)
        else:
            warnings.warn(f'`MODELS` is not defined in `{DEFAULT_SCOPE}.models'
                          f'.builder`! Using MME `MODELS` registry instead!')

    return MODELS.build(cfg)


def build_mutable(cfg):
    return MUTABLES.build(cfg)


def build_mutator(cfg):
    return MUTATORS.build(cfg)


def build_architecutre(cfg):
    return ARCHITECTURES.build(cfg)


def build_op(cfg):
    return OPS.build(cfg)


def build_quantizer(cfg):
    return QUANTIZER.build(cfg)


def build_observer(cfg):
    return OBSERVERS.build(cfg)


def build_fake_quantization(cfg):
    return FAKE_QUANTIZATION.build(cfg)


def build_generator(cfg):
    return GENERATORS.build(cfg)


def build_model_wrapper(cfg):
    import_used_modules(cfg, 'models')

    from mme.utils import DEFAULT_SCOPE
    if DEFAULT_SCOPE:
        registry = replace_default_registry(DEFAULT_SCOPE, 'MODEL_WRAPPERS',
                                            'models.builder')
        if registry:
            return registry.build(cfg)
        else:
            warnings.warn(
                f'`MODEL_WRAPPERS` is not defined in `{DEFAULT_SCOPE}.models'
                f'.builder`! Using MME `MODEL_WRAPPERS` registry instead!')

    return MODEL_WRAPPERS.build(cfg)
