# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os

from mmcv.utils import get_logger
from mmcv.utils.logging import logger_initialized

from .path import trace_up


def get_root_logger(log_file=None,
                    log_level=logging.INFO,
                    auto_find_file=False):
    """Get root logger.

    If auto_find_file, requires environment variable `task_name`.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        auto_find_file (bool, optional): Whether to use the log file in HPO.
            Defaults to False.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    if not log_file and auto_find_file:
        task_name = os.environ.get('task_name', '')
        try:
            root_path = trace_up('.search-run')
        except FileNotFoundError:
            logger = get_logger(
                name='vitmvt', log_file=log_file, log_level=log_level)
            return logger

        log_path = os.path.join(root_path, 'logs')
        if task_name != '':
            file_name = os.path.join(log_path, '{}.log'.format(task_name))
            logger = get_logger(
                name=task_name,
                log_file=file_name,
                log_level=log_level,
                file_mode='a')
        else:
            file_name = os.path.join(log_path, 'vitmvt_log.log')
            logger = get_logger(
                name='vitmvt', log_file=file_name, log_level=log_level)
    else:
        logger = get_logger(name='vitmvt', log_file=log_file, log_level=log_level)

    return logger



def get_multitask_root_logger(name='vitmvt',
                              log_file=None,
                              log_level=logging.INFO,
                              task_rank=0,
                              task_name=None,
                              file_mode='w'):
    """Initialize and get a multitask logger by name.

    This function will init loggers for every task in multitask training.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process task_rank is 0, a
    FileHandler will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            task_rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        task_rank (int): Task rank.
        task_name (str): Task name. Will appear in logger format string.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    # only task_rank 0 will add a FileHandler
    if task_rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        f'%(asctime)s - %(name)s - task_{task_name} - %(levelname)s'
        ' - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if task_rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger
