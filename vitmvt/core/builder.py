from mmcv.runner import HOOKS as MMCV_HOOKS
from mmcv.runner import OPTIMIZER_BUILDERS as MMCV_OPTIMIZER_BUILDERS
from mmcv.runner import OPTIMIZERS as MMCV_OPTIMIZER
from mmcv.runner import RUNNERS as MMCV_RUNNERS
from mmcv.utils import Registry


def build_model_searcher(cfg, default_args=None):
    return MODEL_SEARCHERS.build(cfg, default_args=default_args)


HOOKS = Registry('hook', parent=MMCV_HOOKS)
MODEL_SEARCHERS = Registry('model_searcher')
RUNNERS = Registry('runner', parent=MMCV_RUNNERS)
OPTIMIZERS = Registry('optimizer', parent=MMCV_OPTIMIZER)
OPTIMIZER_BUILDERS = Registry(
    'optimizer builder', parent=MMCV_OPTIMIZER_BUILDERS)
