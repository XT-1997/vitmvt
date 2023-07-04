import bisect
from typing import Dict, Mapping, Sequence

from torch.utils.data import Dataset

from vitmvt.datasets.builder import DATASETS, build_dataset


@DATASETS.register_module()
class MultiSourceDataset(Dataset):
    """This dataset only for train.

    It supports multi-source datasets, which can contain different pipelines
    and output different keys. It will automatically prefix the output keys,
    and the prefix is set in dataset config, default is ['dataset_0',
    'dataset_1', ...]

    Args:
        datasets (Sequence): Configurations of datasets.

    Examples:
        >>> dataset_1 = dict(type='mmcls.CIFAR10', prefix='cls', ...)
        >>> dataset_2 = dict(type='mmdet.COCO', prefix='det', ...)
        >>> dataset = MultiSourceDataset([dataset_1, dataset_2])
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            length = len(e)
            r.append(length + s)
            s += length
        return r

    def __init__(self, datasets):
        assert isinstance(
            datasets,
            Sequence), f'datasets should be Sequence, but got {datasets}'
        if isinstance(datasets[0], Mapping):
            self.prefixes = [
                dataset.pop('prefix')
                if 'prefix' in dataset else f'dataset_{i}'
                for i, dataset in enumerate(datasets)
            ]
            self.datasets = [build_dataset(cfg) for cfg in datasets]
        else:
            self.prefixes = [f'dataset_{i}' for i, _ in enumerate(datasets)]
            self.datasets = datasets
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self._convert_keys(dataset_idx,
                                  self.datasets[dataset_idx][sample_idx])

    def _convert_keys(self, dataset_idx, results):
        """Prefix the output keys."""
        assert isinstance(
            results, Dict
        ), f'Data pipeline result must be dict, but get {type(results)}'
        new_result = dict()
        for k, v in results.items():
            new_result[f'{self.prefixes[dataset_idx]}.{k}'] = v
        return new_result
