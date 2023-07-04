# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import Sampler

from ..builder import SAMPLERS


@SAMPLERS.register_module()
class SliceSampler(Sampler):
    """It is designed for `IterationBased` runner and `ConcatDataset` and
    yields a mini-batch indices each time.

    Args:
        dataset (object): The dataset.
        num_replicas (int, optional): Number of processes participating in
            distributed training. Default: None.
        rank (int, optional): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
        slice_size (list): Return size from each datasets of ConcatDataset.
    """  # noqa: W605

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 shuffle=True,
                 slice_size=None):
        assert isinstance(dataset, ConcatDataset)
        assert (slice_size is not None
                and len(slice_size) == len(dataset.datasets))
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.seed = seed if seed is not None else 0
        self.shuffle = shuffle
        self.slice_size = slice_size

        self.size = len(dataset)
        self.indices = self._get_indices()

    def _infinite_indices(self, start, size):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        epoch = 0
        while True:
            if self.shuffle:
                g.manual_seed(self.seed + epoch)
                indices = torch.randperm(size, generator=g).tolist()
            else:
                indices = torch.arange(size).tolist()

            indices = indices[self.rank::self.num_replicas]
            for idx in indices:
                yield start + idx
            epoch += 1

    def _get_indices(self):
        start_list = [0] + self.dataset.cumulative_sizes[:-1]
        size_list = [len(d) for d in self.dataset.datasets]
        indices = []
        for start, size in zip(start_list, size_list):
            indices.append(self._infinite_indices(start, size))
        return indices

    def __iter__(self):
        while True:
            for i, size in enumerate(self.slice_size):
                for j in range(size):
                    yield next(self.indices[i])

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        """Not supported in `IterationBased` runner."""
        raise NotImplementedError
