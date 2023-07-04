import hashlib
import time
import warnings
from datetime import datetime
from importlib import import_module
from mmcv.fileio import dump
from vitmvt.utils import get_root_logger
import glob
import os.path as osp
import numpy as np
import timeout_decorator
import torch
from mmcv.runner import get_dist_info
from torch import distributed as dist


logger = get_root_logger(auto_find_file=True)


def get_current_time(fmt='%Y/%m/%d %H:%M:%S.%f'):
    return datetime.now().strftime(fmt)


def strftime_to_datetime(strftime, fmt='%Y/%m/%d %H:%M:%S.%f'):
    return datetime.strptime(strftime, fmt)


def strftime_to_timestamp(strftime, fmt='%Y/%m/%d %H:%M:%S.%f'):
    stamp_array = datetime.strptime(strftime, fmt)
    stamp = int(time.mktime(stamp_array.timetuple()))
    return stamp


def timedelta_to_dhm(td):
    return round(td.days + td.seconds / (3600 * 24), 2),\
        round(td.days * 24 + td.seconds / 3600, 2),\
        round(td.days * 24 * 60 + td.seconds / 60, 2)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def compute_signature(arg_dict):
    return hashlib.md5(dump(arg_dict,
                            file_format='yaml').encode('utf-8')).hexdigest()


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def find_latest_checkpoint(path, suffix='pth'):
    """Find the latest checkpoint from the working directory.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


def timeout_check(dataloader, dataloader_name, logger=None, seconds=300):
    """Dataloader timeout check.

    Args:
        dataloader (:obj:`DataLoader`): Dataloader to be checked.
        dataloader_name (str): The name of Dataloader to be logger.
        logger (:obj:`logging.Logger`): Logger used during checking.
            Default: None
        seconds (int): Optional time limit in seconds or fractions of a second.
            Default: 300
    """

    @timeout_decorator.timeout(seconds)
    def check():
        if logger is not None:
            logger.info(f'Checking {dataloader_name}...')
        _ = next(iter(dataloader))
        if logger is not None:
            logger.info(f'Check {dataloader_name} succeeded.')

    try:
        check()
    except timeout_decorator.TimeoutError:
        raise timeout_decorator.TimeoutError(
            f'{dataloader_name} timed out loading the first piece of data')
