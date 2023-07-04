import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import NORM_LAYERS
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import MUTATORS
from .base_mutable import SliceOp


class _BatchNormNdSlice(_BatchNorm, SliceOp):
    """Sliceable BN slice module.

    Args:
        num_features (int/Int): Same as _BatchNorm.
        eps (float, optional): Same as _BatchNorm. Defaults to 1e-5.
        momentum (float, optional): Same as _BatchNorm. Defaults to 0.1.
        affine (bool, optional): Same as _BatchNorm. Defaults to True.
        track_running_stats (bool, optional): Same as _BatchNorm.
            Defaults to True.
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 key=None,
                 **kwargs):
        # Must initionalize first for multiple inheritance.
        SliceOp.__init__(self, key=key)
        max_num_features = self.get_value(num_features, kind='max')
        super(_BatchNormNdSlice,
              self).__init__(max_num_features, eps, momentum, affine,
                             track_running_stats, **kwargs)
        self.num_features = num_features

    def forward(self, *args, **kwargs):
        """For multiple inheritance, we use the SliceOp's ``forward`` method to
        proxy.

        Note that, each mutiple inheritance mutable should realize this.
        """
        return SliceOp.forward(self, *args, **kwargs)

    def forward_inner(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = \
                    1.0 / int(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        num_features = input.size(1)  # self.get_value(self.num_features)
        if self.track_running_stats:
            running_mean = self.running_mean[:num_features]
            running_var = self.running_var[:num_features]
        else:
            running_mean = None
            running_var = None
        if self.affine:
            weight = self.weight[:num_features]
            bias = self.bias[:num_features]
            # if hasattr(self.weight, 'grad_factor'):
            #     self.weight.grad_factor[:num_features] += 1
            # if hasattr(self.bias, 'grad_factor'):
            #     self.bias.grad_factor[:num_features] += 1
        else:
            weight = None
            bias = None

        return F.batch_norm(input, running_mean, running_var, weight, bias,
                            self.training or not self.track_running_stats,
                            exponential_average_factor, self.eps)

    def _check_input_dim(self, input):
        raise NotImplementedError


@MUTATORS.register_module()
@NORM_LAYERS.register_module()
class BatchNorm2dSlice(_BatchNormNdSlice):
    """BatchNorm 2d slice op."""
    _check_input_dim = nn.BatchNorm2d._check_input_dim

    def export(self, **kwargs):
        num_features = kwargs.get('num_features',
                                  self.get_value(self.num_features))

        export_module = nn.BatchNorm2d(
            num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats)
        export_module.num_batches_tracked = self.num_batches_tracked
        if self.track_running_stats:
            export_module.running_mean.data.copy_(
                self.running_mean.data[:num_features])
            export_module.running_var.data.copy_(
                self.running_var.data[:num_features])
        if self.affine:
            export_module.weight.data.copy_(self.weight.data[:num_features])
            export_module.bias.data.copy_(self.bias.data[:num_features])
        return export_module


@MUTATORS.register_module()
@NORM_LAYERS.register_module()
class BatchNorm3dSlice(_BatchNormNdSlice):
    """BatchNorm 3d slice op."""
    _check_input_dim = nn.BatchNorm3d._check_input_dim

    def export(self, **kwargs):
        num_features = kwargs.get('num_features',
                                  self.get_value(self.num_features))

        export_module = nn.BatchNorm3d(
            num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats)
        export_module.num_batches_tracked = self.num_batches_tracked
        if self.track_running_stats:
            export_module.running_mean.data.copy_(
                self.running_mean.data[:num_features])
            export_module.running_var.data.copy_(
                self.running_var.data[:num_features])
        if self.affine:
            export_module.weight.data.copy_(self.weight.data[:num_features])
            export_module.bias.data.copy_(self.bias.data[:num_features])
        return export_module


@MUTATORS.register_module()
@NORM_LAYERS.register_module()
class SyncBatchNormSlice(_BatchNormNdSlice):
    """SyncBatchNorm slice op."""
    _check_input_dim = nn.SyncBatchNorm._check_input_dim
    # _specify_ddp_gpu_num = nn.SyncBatchNorm._specify_ddp_gpu_num

    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 frozen=False,
                 process_group=None,
                 sync_stats=False,
                 key=None,
                 **kwargs):

        super(SyncBatchNormSlice, self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            key=key,
            **kwargs)
        self.process_group = process_group
        # compatiable for parrots and pytorch
        self.frozen = frozen
        self.sync_stats = sync_stats
        self.ddp_gpu_size = None

    def forward_inner(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = (
                        1.0 / float(self.num_batches_tracked))
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        num_features = input.size(1)  # self.get_value(self.num_features)
        if self.track_running_stats:
            running_mean = self.running_mean[:num_features]
            running_var = self.running_var[:num_features]
        else:
            running_mean = None
            running_var = None
        if self.affine:
            weight = self.weight[:num_features]
            bias = self.bias[:num_features]
        else:
            weight = None
            bias = None

        need_sync = self.training or not self.track_running_stats
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        if need_sync:
            if torch.__version__ == 'parrots':
                from parrots.nn.functional import sync_batch_norm
                return sync_batch_norm(
                    input,
                    running_mean,
                    running_var,
                    weight,
                    bias,
                    self.training,
                    momentum=exponential_average_factor,
                    eps=self.eps,
                    frozen=self.frozen,
                    group=self.process_group,
                    sync_stats=self.sync_stats)
            else:
                # if not self.ddp_gpu_size:
                #     raise AttributeError(
                #         'SyncBatchNormSlice is only supported ' +
                #         'within torch.nn.parallel.DistributedDataParallel')
                from torch.nn.modules.batchnorm import sync_batch_norm
                return sync_batch_norm.apply(input, weight, bias, running_mean,
                                             running_var, self.eps,
                                             exponential_average_factor,
                                             process_group, world_size)

        else:
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                weight,
                bias,
                training=False,
                momentum=exponential_average_factor,
                eps=self.eps)

    def export(self, **kwargs):
        num_features = kwargs.get('num_features',
                                  self.get_value(self.num_features))

        export_module = nn.SyncBatchNorm(
            num_features=num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            process_group=self.process_group)

        export_module.num_batches_tracked = self.num_batches_tracked
        if self.track_running_stats:
            export_module.running_mean.data.copy_(
                self.running_mean.data[:num_features])
            export_module.running_var.data.copy_(
                self.running_var.data[:num_features])
        if self.affine:
            export_module.weight.data.copy_(self.weight.data[:num_features])
            export_module.bias.data.copy_(self.bias.data[:num_features])

        return export_module
