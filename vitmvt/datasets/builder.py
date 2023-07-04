import random
from functools import partial

import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, digit_version, import_modules_from_strings
from mmcv.utils.parrots_wrapper import DataLoader, PoolDataLoader

from vitmvt.utils import add_children_to_registry, import_used_modules
from vitmvt.utils.collate import collate
from vitmvt.utils.logger import get_root_logger

try:
    # Use petrel Dataloader when torch version < 1.7.0.
    # See this link for details on petrel Dataloader,
    # https://confluence.sensetime.com/display/PlatformSupport/Petrel-OSS+Python+SDK  # noqa: E501
    from petrel_client.utils.data import DataLoader as AvoidDeadLockDataLoader
except ImportError:
    AvoidDeadLockDataLoader = None

ROOT_DATASETS = Registry('dataset', scope='root')
DATASETS = Registry('dataset', parent=ROOT_DATASETS)

ROOT_PIPELINES = Registry('pipeline', scope='root')
PIPELINES = Registry('pipeline', parent=ROOT_PIPELINES)

ROOT_SAMPLERS = Registry('sampler', scope='root')
SAMPLERS = Registry('sampler', parent=ROOT_SAMPLERS)


def build_dataset(cfg, default_args=None):
    """
    API wrapper of build_dataset. Call different build func from
    different repo. Need to design a unified dataset API in
    the future.
    Args:
        cfg: dataset config
        default_args: None

    Returns: Dataset

    """
    # TODO: use unified dataset API
    if 'dataset' in cfg:
        dataset_type = cfg['dataset']['type']
        # 'pipeline' may not in cfg when use multi source datasets.
        pipeline_cfg = cfg['dataset'].get('pipeline', [])
    else:
        dataset_type = cfg['type']
        # 'pipeline' may not in cfg when use multi source datasets.
        pipeline_cfg = cfg.get('pipeline', [])

    # TODO: remove after adding root registry in mmcv
    # No root registry of DATASETS in mmcv,
    # need to dfs config to import all used datasets
    modules = import_used_modules(pipeline_cfg, 'datasets')
    add_children_to_registry(ROOT_DATASETS, 'DATASETS', modules)

    # No root registry of PIPELINES in mmcv,
    # need to dfs config to import all used pipelines
    modules = import_used_modules(pipeline_cfg, 'datasets')
    add_children_to_registry(ROOT_PIPELINES, 'PIPELINES', modules)

    if '.' not in dataset_type:
        # use default scope when type is not defined
        from vitmvt.utils import DEFAULT_SCOPE
        if DEFAULT_SCOPE:
            # directly call default scope's build function
            build_func = import_modules_from_strings(DEFAULT_SCOPE +
                                                     '.datasets').build_dataset
            return build_func(cfg=cfg, default_args=default_args)
        else:
            raise AssertionError('When default scope is not defined, '
                                 'scope is necessary in dataset type!')

    scope = dataset_type.split('.')[0]
    if scope == 'vitmvt':
        return DATASETS.build(cfg=cfg, default_args=default_args)
    else:
        pac_name = scope + '.datasets'
        build_func = import_modules_from_strings(pac_name).build_dataset
        return build_func(cfg=cfg, default_args=default_args)


def build_sampler(cfg, default_args=None):
    if cfg is None:
        return None
    else:
        return SAMPLERS.build(cfg, default_args=default_args)


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     round_up=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     sampler_cfg=None,
                     dataloader_type=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        round_up (bool): Whether to round up the length of dataset by adding
            extra samples to make it evenly divisible. Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        sampler_cfg (dict, optional): Config of the sampler. Default: None.
        dataloader_type (str): Type of dataloader. Default: 'PoolDataLoader'
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()

    # Custom sampler logic
    if sampler_cfg:
        # shuffle=False when val and test
        sampler_cfg.update(shuffle=shuffle)
        sampler = build_sampler(
            sampler_cfg,
            default_args=dict(
                dataset=dataset, num_replicas=world_size, rank=rank))
    # Default sampler logic
    elif dist:
        from .samplers import DistributedSampler
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle, round_up=round_up)
    else:
        sampler = None

    # If sampler exists, turn off dataloader shuffle
    if sampler is not None:
        shuffle = False

    # set batch size and workers
    if dist:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    dataloader = _get_dataloader(dataloader_type)

    data_loader = dataloader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)
    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _get_dataloader(dataloader_type):
    """Get the dataloader class.

    Note:
        When the ``dataloader_type`` is None, the highest performance
        dataloader will be automatically obtained according to the environment.
    """
    if dataloader_type is not None:
        assert dataloader_type in (
            'DataLoader', 'PoolDataLoader', 'AvoidDeadLockDataLoader'
        ), f'unsupported dataloader {dataloader_type}'
        dataloader = eval(dataloader_type)
        if dataloader is None:
            raise ValueError(f'{dataloader_type} is not available, '
                             'please install or update petrel sdk.')
        return dataloader
    logger = get_root_logger()
    logger.info(
        'The dataloader_type is none, so the dataloader is set automatically.')
    if torch.__version__ == 'parrots':
        logger.info('Env: Parrots\tDataloader Type: PoolDataloader')
        dataloader = PoolDataLoader
    elif digit_version(torch.__version__) < digit_version(
            '1.7.0') and AvoidDeadLockDataLoader is not None:
        logger.info(f'Env: torch{torch.__version__}\tDataloader Type:'
                    'AvoidDeadLockDataLoader')
        dataloader = AvoidDeadLockDataLoader
    else:
        logger.info(
            f'Env: torch{torch.__version__}\tDataloader Type: DataLoader')
        dataloader = DataLoader
    return dataloader
