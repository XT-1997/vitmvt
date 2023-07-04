# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import mmcv
import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class BaseDataset(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows.

    .. code-block:: none
        data_info = {
            ## original dataset dict
            'filename': str
            'height': int
            'width': int
            'metadata': {
                'instance_tasks': {
                    'task_1': dict
                    ...
                    'task_m': dict
                },
            },
            'instances': [
                {
                    'bbox': list[float]
                    'bbox_label': int
                    'mask': list[list[float]] or dict
                    'keypoints': list[float]
                    'extra_anns': list[int]
                    ...
                },
                ...
            ],
            'seg_map': str
            'segments_info': list[dict]
            'img_label': <int>, the label of the image,
                         for image classification
            'img_task1_label': <int> , the label of the image,
                               for image classification task 1
            'img_task2_label': <int>

            ## updated by LoadAnnotations pipeline
            # fields that specify and organize data of similar types
            'img_fields': ['img'],
            'bbox_fields': ['gt_bboxes', 'gt_bboxes_ignore'],
            'seg_fields': ['gt_semantic_seg'],
            'mask_fields': ['gt_masks'],
            'keypoints_fields': ['gt_keypoints'],
            # ground truth to be used for training
            'img': np.ndarray (H, W, 3)
            'gt_bboxes': np.ndarray (n, 4) in (x1, y1, x2, y2) order.
            'gt_labels': np.ndarray (n, ),
            'gt_bboxes_ignore': np.ndarray (k, 4), (optional field)
            'gt_labels_ignore': np.ndarray (k, 4) (optional field)
            'gt_masks': mask structures, information of instance masks,
                        having different keys for instance seeg and panoptic seg,
                        respectively
            'gt_keypoints': np.ndarray in (N, NK, 4) or (N, NK, 3) ,
                            in (x, y, z, visibility) order, NK = number
                            of keypoints per object
            'relationship': np.ndarray (NA, 3), NA is the number of
                            relationship in this image
            'gt_semantic_seg': np.ndarray (H, W),
            'gt_task1_labels': np.ndarray (n, ), the label of task1 of n objects
            'gt_task2_labels': np.ndarray (n, ), the label of task1 of n objects
            ...
            'gt_taskm_labels': np.ndarray (n, ), the label of task1 of n objects
        }

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        data_root (str, optional): Data root for ``ann_file`` and
            ``data_prefix`` if specified.
        data_prefix (dict): Path prefix for filenames in the annotation file.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_cfg (dict | optional): A dict of filtering strategy, set the
            data filtering criteria. The key `img_size` is a tuple of (h, w),
            image with height or width smaller than this will be filtered.
            Default None.
    """ # noqa

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 data_prefix=dict(img=None, seg_map=None),
                 test_mode=False,
                 filter_cfg=None):
        self.ann_file = ann_file
        self.data_root = data_root
        self.data_prefix = data_prefix
        self.test_mode = test_mode

        # join paths
        self._join_prefix()

        # load metadata and annotations
        self.metadata, self.data_infos = self.load_annotations(self.ann_file)

        # filter images too small and containing no annotations
        if not test_mode and filter_cfg is not None:
            valid_inds = self._filter_imgs(**filter_cfg)
            self.data_infos = [self.data_infos[i] for i in valid_inds]

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def _join_prefix(self):
        """Join data_root and data_prefix."""
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            for data_key, prefix in self.data_prefix.items():
                if not (prefix is None or osp.isabs(prefix)):
                    self.data_prefix[data_key] = osp.join(
                        self.data_root, prefix)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        ann = mmcv.load(ann_file)
        return ann['metadata'], ann['data_infos']

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]

    def num_instances(self):
        return sum(
            [len(self.get_ann_info(i)['instances']) for i in range(len(self))])

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        for key, prefix in self.data_prefix.items():
            results[f'{key}_prefix'] = prefix
        results['img_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['keypoints_fields'] = []

    def _filter_imgs(self, img_size=(0, 0)):
        """Filter images.

        Args:
            img_size (tuple[int]): Minimal image height and width.
                Images smaller than this size will be filtered.
                Default: (0, 0).
        """
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if img_info['height'] > img_size[0] and img_info[
                    'width'] > img_size[1]:
                valid_inds.append(i)
        return valid_inds

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Placeholder to format result to dataset specific output."""
        return results

    def evaluate(self, results, **kwargs):
        """Placeholder of evaluating the results."""
        warnings.warn('BaseDataset does not support evaluate.'
                      'Please overwrite this function.')

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                  f'with number of images {len(self)}, '
                  f'and instance counts: {self.num_instances()}.\n')
        return result
