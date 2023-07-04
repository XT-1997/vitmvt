import os
import shutil
import tarfile
import time

from mmcv.fileio import dump, load

from vitmvt.utils import get_root_logger
from vitmvt.utils.path import trace_up

logger = get_root_logger(auto_find_file=True)
# PATH = os.path.join(os.environ.get('PWD'), '.search-run/tasks')
try:
    PATH = trace_up('.search-run')
except Exception:
    PATH = os.environ.get('PWD')


def export_task(task_name, tar_path=None, logger=None):
    if not logger:
        logger = get_root_logger(auto_find_file=True)
    task_path = os.path.join(PATH, 'tasks', task_name)
    current_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    if not os.path.exists(task_path):
        logger.error('task {} is not exist!'.format(task_name))
        exit(1)
    if not tar_path:
        tar_save_dir = os.path.join(os.environ.get('PWD'), 'archive')
        if not os.path.exists(tar_save_dir):
            os.makedirs(tar_save_dir)
        tar_path = os.path.join(tar_save_dir,
                                '{}.{}.tar.gz'.format(task_name, current_time))
    try:
        tar = tarfile.open(tar_path, 'w:gz')
    except FileNotFoundError:
        logger.info('No such file or directory: {}'.format(tar_path))
        exit(1)
    for root, dirs, files in os.walk(task_path):
        root_ = os.path.relpath(root, start=os.environ.get('PWD'))
        for file in files:
            real_path = os.path.join(root_, file)
            tar.add(real_path)
    # store <task_name>.log and summary.yaml
    log_path = os.path.join(PATH, 'logs/{}.log'.format(task_name))
    summary_path = os.path.join(PATH, 'summary.yaml')
    if os.path.exists(log_path):
        log_path = os.path.relpath(log_path, start=os.environ.get('PWD'))
        tar.add(log_path)
    if os.path.exists(summary_path):
        summary_info = load(summary_path, file_format='yaml')
        task_summary_info = summary_info[task_name]
        task_summary_info = {task_name: task_summary_info}
        task_summary_path = summary_path.replace(
            'summary.yaml', 'summary_{}.yaml'.format(task_name))
        dump(task_summary_info, task_summary_path, file_format='yaml')
        task_summary_path = os.path.relpath(
            task_summary_path, start=os.environ.get('PWD'))
        tar.add(task_summary_path)
        os.remove(task_summary_path)
    tar.close()
    logger.info('export {} successfully! save path: {}'.format(
        task_name, tar_path))


def import_task(task_name=None, tar_path=None, force=False, logger=None):
    if not logger:
        logger = get_root_logger(auto_find_file=True)
    if not tar_path:
        logger.error('Please specify tar_path')
        exit(1)
    if not os.path.isfile(tar_path):
        logger.error('tar-path must be a file.')
        exit(1)
    if not task_name:
        tar = tarfile.open(tar_path, 'r:gz')
        file_names = tar.getnames()
        for file_name in file_names:
            if 'summary' in file_name:
                task_name = file_name.split('_')[-1].split('.')[0]
                break
    if not task_name:
        logger.error('Can\'t find task name')
        exit(1)
    task_path = os.path.join(PATH, 'tasks', task_name)
    if os.path.exists(task_path):
        while not force:
            answer = input('Will overwrite records of task {}, do you '
                           'wish to continue?(y/n)'.format(task_name))
            if answer == 'y':
                break
            elif answer == 'n':
                exit()
    summary_path = os.path.join(PATH, 'summary.yaml')
    summary_info = load(summary_path, file_format='yaml')
    if task_name in summary_info.keys():
        while True:
            answer = input('Will overwrite summary.yaml records of '
                           'task info {} do you wish to continue?'
                           '(y/n)'.format(task_name))
            if answer == 'y':
                break
            elif answer == 'n':
                exit()
    if os.path.exists(task_path):
        shutil.rmtree(task_path)
    try:
        tar = tarfile.open(tar_path, 'r:gz')
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, os.environ.get('PWD'))
        # restore summary.yaml
        task_summary_path = os.path.join(PATH,
                                         'summary_{}.yaml'.format(task_name))
        if os.path.exists(task_summary_path) and os.path.exists(summary_path):
            task_summary_info = load(task_summary_path, file_format='yaml')
            summary_info.update(task_summary_info)
            dump(summary_info, summary_path)
            os.remove(task_summary_path)
        elif (os.path.exists(task_summary_path)
              and not os.path.exists(summary_path)):
            os.rename(task_summary_path, summary_path)
        tar.close()
        logger.info('import {} successfully! load from: {}'.format(
            task_name, tar_path))
    except Exception as e:
        logger.error('error: {}'.format(e))
        exit(1)
