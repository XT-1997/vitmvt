# Copyright (c) OpenMMLab. All rights reserved.
import warnings


def config_compat(cfg):
    """In order to use the configs from OpenMMLab open source codebases, some
    fields in the config need to be set."""

    # to avoid some config do not have runner
    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'mmcv.EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    if cfg.get('default_scope') == 'mmdet':
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        from mmdet.datasets import replace_ImageToTensor
        cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    if 'demo_pipeline' not in cfg:
        cfg.demo_pipeline = cfg.test_pipeline

    # to call test api from default scope
    if 'test_setting' not in cfg:
        test_setting_cfg = {
            'repo': cfg.default_scope,
            'single_gpu_test': {
                'show': False
            },
            'multi_gpu_test': {},
        }
        cfg.test_setting = test_setting_cfg

    # switch to mme eval hook
    if 'evaluation' not in cfg:
        cfg.evaluation = {
            'by_epoch': cfg.runner['type'].split('.')[-1] != 'IterBasedRunner'
        }
    cfg.evaluation.setdefault('type', 'mme.EvalHook')
    if 'dataset' not in cfg.evaluation:
        if 'val' not in cfg.data:
            raise KeyError('if enabling validation, the `val` key should '
                           f'be set in `cfg.data`, but got {cfg.data}.')
        cfg.evaluation['dataset'] = cfg.data.val
    if 'dataloader' not in cfg.evaluation:
        if 'samples_per_gpu' not in cfg.data:
            raise KeyError('if enabling validation, the `samples_per_gpu`'
                           ' key should be set in `cfg.data`, but got '
                           f'{cfg.data}.')
        if 'workers_per_gpu' not in cfg.data:
            raise KeyError('if enabling validation, the `workers_per_gpu`'
                           ' key should be set in `cfg.data`, but got '
                           f'{cfg.data}.')
        data_loader_cfg = {
            'samples_per_gpu': cfg.data.samples_per_gpu,
            'workers_per_gpu': cfg.data.workers_per_gpu
        }
        cfg.evaluation.dataloader = data_loader_cfg
    if 'test_setting' not in cfg.evaluation:
        cfg.evaluation.test_setting = cfg.test_setting

    return cfg


def get_dataloader_cfg(cfg, loader_type='train_loader'):
    """Check and set the default dataloader parameter."""
    dataloader_cfg = dict()
    if loader_type in cfg.data:
        dataloader_cfg.update(**cfg.data.get(loader_type))
    else:
        dataloader_cfg.update(
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu)
    return dataloader_cfg
