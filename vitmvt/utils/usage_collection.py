#!/usr/bin/python

import atexit
import os
import sys
import time
import traceback
from datetime import datetime

import requests
import torch.distributed as dist
from easydict import EasyDict
from requests.auth import HTTPBasicAuth

import vitmvt

process_begin_time = time.time()


def get_user_info():
    username = os.getenv('USER')
    hostname = os.getenv('HOSTNAME')
    if hostname in [None, '']:
        hostname = os.getenv('HOST')
    time = datetime.now().strftime('%Y/%m/%d, %H:%M:%S')
    road = vitmvt.__file__[:-15]
    # TODO(@liujiahao):
    # git_hash = os.popen('git --git-dir={}.git '
    #                     'rev-parse HEAD'.format(road)).read()

    try:
        slurm_jobid = os.environ['SLURM_JOB_ID']
    except KeyError:
        pass

    workdir = os.getcwd()
    cluster = os.getenv('HOSTNAME')
    vitmvt_version = vitmvt.__version__

    return EasyDict(locals())


class _UsageCollector:
    auth = ['vitmvt', 'b3881301956c8b71f92acec7ca6f147f']
    pb_url = 'http://pb.parrots.sensetime.com/pb.gif'
    algorithm_info = None
    algo = None
    function = None
    model_class = None
    downstream_codebase = None
    pid_flag = None
    pid = []

    def __init__(self):
        pass

    def check_rank_sanity(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        return rank == 0

    def update_algo_usage(self, algorithm_info, model_class):
        self.algorithm_info = algorithm_info
        self.model_class = model_class
        if self.check_rank_sanity():
            self.downstream_codebase = str(model_class).split("'")[1].split(
                '.')[0] if model_class != 'HPO' else 'HPO'
            self.algo = str(algorithm_info).split("'")[1].split('.')[-1]
            self.function = str(algorithm_info).split("'")[1].split('.')[-3]

            params = {
                **get_user_info(),
                'algo': self.algo,
                'function': self.function,
                'downstream_codebase': self.downstream_codebase,
            }
            requests.get(  # noqa: F841
                self.pb_url,
                auth=HTTPBasicAuth(self.auth[0], self.auth[1]),
                params={**params})

    def run_time(self, run_time):
        if self.check_rank_sanity():
            params = {
                **get_user_info(), 'run_time': run_time,
                'algo': self.algo,
                'function': self.function,
                'downstream_codebase': self.downstream_codebase
            }
            requests.get(
                self.pb_url,
                auth=HTTPBasicAuth(self.auth[0], self.auth[1]),
                params={**params})

    def update_error(self, lines, runtime, exception_type):
        if self.check_rank_sanity():
            exception_info = '\n'.join(lines)
            params = {
                **get_user_info(), 'exception_info': exception_info,
                'run_time': runtime,
                'algo': self.algo,
                'function': self.function,
                'downstream_codebase': self.downstream_codebase,
                'exception_type': exception_type
            }
            requests.get(  # noqa: F841
                self.pb_url,
                auth=HTTPBasicAuth(self.auth[0], self.auth[1]),
                params={**params})

    def pid_number(self, pid_flag):
        if self.check_rank_sanity():
            params = {**get_user_info(), 'pid': pid_flag}
            requests.get(
                self.pb_url,
                auth=HTTPBasicAuth(self.auth[0], self.auth[1]),
                params={**params})


USAGE_COLLECTOR = _UsageCollector()


# log uncaught exceptions
def log_exceptions(type, value, tb):
    res = []
    for line in traceback.TracebackException(type, value,
                                             tb).format(chain=True):
        res.append(line)
    exception_type = type
    res = res[:-1]
    end_time = time.time()
    run_time = end_time - process_begin_time
    USAGE_COLLECTOR.update_error(res, run_time, exception_type)

    USAGE_COLLECTOR.pid_number(pid_flag=6)

    sys.__excepthook__(type, value, tb)  # calls default excepthook


sys.excepthook = log_exceptions


def exit_handler():
    end_time = time.time()
    run_time = end_time - process_begin_time
    USAGE_COLLECTOR.pid_number(pid_flag=7)
    USAGE_COLLECTOR.run_time(run_time)


atexit.register(exit_handler)
