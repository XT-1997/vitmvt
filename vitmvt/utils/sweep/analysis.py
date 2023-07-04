"""Sweep analysis functions."""
import json
import os

import numpy as np
from mmcv import Config


def load_sweep_mm(sweep_file):
    """Loads sweep data from a file."""
    with open(sweep_file, 'r') as f:
        sweep = json.load(f)
    return sweep


def describe_sweep_mm(sweep, reverse=False):
    """Generate a string description of sweep."""

    keys = [
        'accuracy_top-1/max', 'capacity/max', 'flops/max', 'cfg_file',
        'samples_per_gpu', 'max_epochs'
    ]
    formats = [
        'top-1-err={:.2f}', 'capacity={:.2f}', 'flops={:.2f}', '{}',
        'samples_per_gpu={}', 'max_epochs={}'
    ]
    vals = [get_mm_vals(sweep, key) for key in keys]
    vals[3] = [v.split('/')[-1] for v in vals[3]]
    desc = [' | '.join(formats).format(*val) for val in zip(*vals)]
    desc = [s for _, s in sorted(zip(vals[0], desc), reverse=reverse)]
    return '\n'.join(desc)


def get_mm_vals(sweep, metric, model=None):
    """Gets values for given metric."""

    compound_key = metric

    vals = []
    if metric == 'cfg_file':
        vals = list(sweep.keys())
    elif compound_key in [
            'accuracy_top-1', 'accuracy_top-1/max', 'accuracy_top-1/min'
    ]:
        for key, value in sweep.items():
            # print(key, value['10'].keys())
            if compound_key in value[str(len(value))]:
                vals.append(100 - value[str(len(value))][compound_key][0])
                # print(key,value[str(len(value))].keys())
            else:
                vals.append(100 - value[str(len(value) - 1)][compound_key][0])
    elif compound_key == 'capacity/max':
        for key, value in sweep.items():
            if compound_key in value[str(len(value))]:
                vals.append(value[str(len(value))][compound_key][0])
            else:
                vals.append(value[str(len(value) - 1)][compound_key][0])
    elif compound_key == 'flops/max':
        for key, value in sweep.items():
            if compound_key in value[str(len(value))]:
                vals.append(value[str(len(value))][compound_key][0])
            else:
                vals.append(value[str(len(value) - 1)][compound_key][0])
    elif metric == 'samples_per_gpu':
        for py_cfg in sweep.keys():
            cfg = Config.fromfile(py_cfg)
            vals.append(cfg.data.samples_per_gpu)
    elif metric == 'max_epochs':
        for py_cfg in sweep.keys():
            cfg = Config.fromfile(py_cfg)
            vals.append(cfg.runner.max_epochs)
    if model:
        tmp_folder = os.path.join(model.ROOT_DIR, 'tasks', model.TASK_NAME,
                                  'tmp')
        cfg_list = list(sweep.keys())
        yaml_list = []
        for py_cfg in cfg_list:
            cfg = Config.fromfile(py_cfg)
            yaml_list.append(
                os.path.join(tmp_folder,
                             cfg.algorithm.mutable_cfg.split('/')[-1]))
            cfg = Config.fromfile(yaml_list[-1])
            vals.append(cfg.get(metric, None))

    return vals


def get_filters_mm(sweep, metrics, alpha=5, sample=0.25, b=2500):
    """Use empirical bootstrap to estimate filter ranges per metric for
    errors."""
    assert len(sweep), 'Sweep cannot be empty.'
    errs = np.array(get_mm_vals(sweep, 'accuracy_top-1/max'))
    n, b, filters = len(errs), int(b), {}
    percentiles = [alpha / 2, 50, 100 - alpha / 2]
    n_sample = int(sample) if sample > 1 else max(1, int(n * sample))
    samples = [np.random.choice(n, n_sample) for _ in range(b)]
    samples = [s[np.argmin(errs[s])] for s in samples]
    for metric in metrics:
        vals = np.array(get_mm_vals(sweep, metric))
        vals = [vals[s] for s in samples]
        v_min, v_mid, v_max = tuple(np.percentile(vals, percentiles))
        filters[metric] = [v_min, v_mid, v_max]
    return filters


def apply_filters_mm(sweep, filters, model=None):
    """Filter sweep according to dict of filters of form.

    {metric: [min, mid, max]}.
    """
    filters = filters if filters else {}
    for metric, (v_min, _, v_max) in filters.items():
        keep = [v_min <= v <= v_max for v in get_mm_vals(sweep, metric, model)]
        sweep_ = [data for k, data in zip(keep, sweep) if k]

    if type(sweep_) == list:
        re_sweep = dict()
        for idx in sweep_:
            re_sweep[idx] = sweep[idx]
        sweep = re_sweep
    return sweep
