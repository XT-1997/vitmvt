import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import CONV_LAYERS
from torch.nn import Parameter, init

from ..builder import MUTATORS
from .base_mutable import SliceOp


def slice_group_diagonal(tensor, G):
    size = list(tensor.size())
    ndim = len(size)
    D1, D2 = size[:2]
    remain_size = size[2:]
    assert D1 % G == 0
    assert D2 % G == 0
    d1 = D1 // G
    d2 = D2 // G
    tensor = tensor.view(G, d1, G, d2, *remain_size)  # [G, d1, G, d2, ...]
    tensor = tensor.transpose(1, 2)  # [G, G, d1, d2, ...]
    tensor = tensor.diagonal()  # [d1, d2, ..., G]
    permute_dims = [ndim] + list(range(ndim))
    tensor = tensor.permute(permute_dims).contiguous()  # [G, d1, d2, ...]
    tensor = tensor.view(G * d1, d2, *remain_size)  # [G x d1, d2, ...]
    return tensor


class _ConvNdSlice(SliceOp):
    """Sliceable Conv module.

    Args:
        in_channels (int/Int): Same as nn._ConvNd.
        out_channels (int/Int): Same as nn._ConvNd.
        kernel_size (int/Int or tuple[int/Int]): Same as nn._ConvNd.
        stride (int/Int or tuple[int/Int]): Same as nn._ConvNd.
        padding (int/Int or tuple[int/Int] or None): Same as nn._ConvNd.
        dilation (int/Int or tuple[int/Int]): Same as nn._ConvNd.
        groups (int or Int space): Same as nn._ConvNd.
        transposed (bool): If specified as `True`, conv calculation is used
            to construct a callable object of the ``ConvTranspose2dSlice``
            class. Default: False.
        output_padding (): controls the additional size added to one side
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        key (str): Unique id. Defaults to None.
        channel_slice_mode (str): slice mode for channel dim search. Support
            `naive_order` and `group_diagonal`. If `naive_order`, it will slice
            filter channel from low->high. Defaults to `naive_order`. If
            `group_diagonal`, it will reorganize the channel in a diagonal
            unstructal mode.
        kernel_transform_mode (str): transform mode for kernel_size search.
            If `linear_mapping`, it will learn a learn mapping from the largest
            kernel_size to current. If `naive_order`, it will slice filter
            kernal_size from center->sides. Defaults to `naive_order`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 transposed,
                 output_padding,
                 groups,
                 bias,
                 padding_mode,
                 key=None,
                 channel_slice_mode='naive_order',
                 kernel_transform_mode='naive_order',
                 **kwargs):
        super(_ConvNdSlice, self).__init__(key=key)
        self.channel_slice_mode = channel_slice_mode
        self.kernel_transform_mode = kernel_transform_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        # wrap the kernel_size, stride, padding and dilation
        # with ntuple.
        self.kernel_size = self.ntuple(kernel_size)
        self.stride = self.ntuple(stride)
        self.padding = self.ntuple(padding) if padding is not None else None
        self.dilation = self.ntuple(dilation)
        self.transposed = transposed
        if transposed:
            self.in_channels, self.out_channels = \
                self.out_channels, self.in_channels
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        assert padding_mode == 'zeros', 'Only support `zeros` padding_mode'

        all_in_gch = self.in_channels % self.groups
        all_out_gch = self.out_channels % self.groups

        if self.get_value(
                self.in_channels, kind='choices') == self.get_value(
                    self.groups, kind='choices'):
            self.is_depthwise = True
        else:
            if any(self.get_value(all_in_gch, kind='choices')) or any(
                    self.get_value(all_out_gch, kind='choices')):
                raise ValueError(
                    'in_channels {} and out_channels {} is not always divisible \
                    by groups {}'.format(in_channels, out_channels, groups))
            self.is_depthwise = False

        self.max_in_channels = self.get_value(self.in_channels, kind='max')
        self.max_out_channels = self.get_value(self.out_channels, kind='max')
        self.max_kernel_size = [
            self.get_value(ks, kind='max')
            for ks in self.ntuple(self.kernel_size)
        ]
        if self.is_depthwise:
            self.weight = Parameter(
                torch.Tensor(
                    self.max_out_channels,
                    1,
                    *self.max_kernel_size,
                ))
        elif self.channel_slice_mode == 'group_diagonal':
            self.weight = Parameter(
                torch.Tensor(
                    self.max_out_channels,
                    self.max_in_channels,
                    *self.max_kernel_size,
                ))
        elif self.channel_slice_mode == 'naive_order':
            if len(self.get_value(self.groups, kind='choices')) == 1:
                self.max_in_grouch = self.max_in_channels // self.get_value(
                    self.groups, kind='min')
                self.weight = Parameter(
                    torch.Tensor(
                        self.max_out_channels,
                        self.max_in_grouch,
                        *self.max_kernel_size,
                    ))
            else:
                raise ValueError(
                    '"naive_order" slice mode can only support one group')
        else:
            raise ValueError('Unsupported slice mode {}'.format(
                self.channel_slice_mode))
        if self.kernel_transform_mode == 'linear_mapping':
            # register scaling parameters, 7to5_matrix, 5to3_matrix
            scale_params = {}
            ks_set = sorted(
                self.get_value(self.kernel_size, kind='choices')[0])
            for i in range(len(ks_set) - 1):
                ks_small = ks_set[i]
                ks_larger = ks_set[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                scale_params['%s_matrix' % param_name] = Parameter(
                    torch.eye(ks_small**2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)
        elif self.kernel_transform_mode != 'naive_order':
            raise ValueError('Unsupported kernel transform mode {}'.format(
                self.kernel_transform_mode))
        if bias:
            if transposed:
                self.bias = Parameter(torch.Tensor(self.max_in_channels))
            else:
                self.bias = Parameter(torch.Tensor(self.max_out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def curr_attributes(self):
        in_channels = self.get_value(self.in_channels)
        out_channels = self.get_value(self.out_channels)
        groups = self.get_value(self.groups)
        in_group_channels = in_channels // groups
        out_group_channels = out_channels // groups
        kernel_size = tuple(
            self.get_value(ks) for ks in self.ntuple(self.kernel_size))
        stride = tuple(self.get_value(s) for s in self.ntuple(self.stride))
        dilation = tuple(self.get_value(d) for d in self.ntuple(self.dilation))
        output_padding = tuple(
            self.get_value(opd) for opd in self.ntuple(self.output_padding))
        if self.padding is None:
            padding = tuple(
                (ks - 1) // 2 * d for ks, d in zip(kernel_size, dilation))
        else:
            padding = tuple(
                self.get_value(pad) for pad in self.ntuple(self.padding))
        return (
            in_channels,
            out_channels,
            in_group_channels,
            out_group_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        )

    def generate_ks_slice(self, ks_small, ks_larger):
        slicing = []
        for mks, ks in zip(ks_larger, ks_small):
            beg = (mks - ks) // 2
            end = (mks + ks) // 2
            slicing.append(slice(beg, end))
        return slicing

    def slice_params(self, in_channels, out_channels, kernel_size, groups):
        weight = self.weight
        bias = self.bias
        slicing_ks = [slice(None)] * len(kernel_size)
        # slice weight channels
        if self.is_depthwise:
            slicing = [slice(out_channels), slice(None)]
            slicing = tuple(slicing + slicing_ks)
            weight = weight[slicing]
        elif self.channel_slice_mode == 'group_diagonal':
            slicing = [slice(out_channels), slice(in_channels)]
            slicing = tuple(slicing + slicing_ks)
            weight = weight[slicing]
            weight = slice_group_diagonal(weight, groups)
        elif self.channel_slice_mode == 'naive_order':
            max_out_grouch = self.max_out_channels // self.get_value(
                self.groups, kind='min')
            viewing1 = (self.get_value(self.groups, kind='min'),
                        max_out_grouch, self.max_in_grouch) + tuple(
                            self.max_kernel_size)
            in_grouch = in_channels // groups
            out_grouch = out_channels // groups
            slicing1 = [slice(groups), slice(out_grouch), slice(in_grouch)]
            slicing1 = tuple(slicing1 + slicing_ks)
            weight = weight.view(viewing1)[slicing1].contiguous()
            viewing2 = (out_channels, in_grouch) + tuple(self.max_kernel_size)
            weight = weight.view(viewing2)
        else:
            raise ValueError('Unsupported slice type {}'.format(
                self.channel_slice_mode))
        if self.kernel_transform_mode == 'linear_mapping':
            start_filter = weight  # start with max kernel
            ks_set = sorted(
                self.get_value(self.kernel_size, kind='choices')[0])
            for i in range(len(ks_set) - 1, 0, -1):
                src_ks = ks_set[i]
                if src_ks <= kernel_size[0]:
                    break
                target_ks = ks_set[i - 1]
                slicing_ks = self.generate_ks_slice(
                    self.ntuple(target_ks), self.ntuple(src_ks))
                slicing = tuple([slice(None), slice(None)] + slicing_ks)
                _input_filter = start_filter[slicing]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)))
                _input_filter = _input_filter.view(
                    weight.size(0), weight.size(1), target_ks**2)
                _input_filter = _input_filter.view(
                    weight.size(0), weight.size(1), target_ks, target_ks)
                start_filter = _input_filter
            weight = start_filter
        else:
            slicing_ks = self.generate_ks_slice(kernel_size,
                                                self.max_kernel_size)
            slicing = tuple([slice(None), slice(None)] + slicing_ks)
            weight = weight[slicing]
        # slice bias
        if bias is not None:
            bias = bias[:out_channels].contiguous()

        return weight.contiguous(), bias

    def forward_inner(self, x):
        # get current hyperparameters
        (
            in_channels,
            out_channels,
            in_group_channels,
            out_group_channels,
            kernel_size,
            stride,
            padding,
            _,
            dilation,
            groups,
        ) = self.curr_attributes()
        weight, bias = self.slice_params(in_channels, out_channels,
                                         kernel_size, groups)
        return self.conv_func(x, weight, bias, stride, padding, dilation,
                              groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        s += ', padding={padding}, dilation={dilation}'
        s += ', output_padding={output_padding}, groups={groups}'
        s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def ntuple(self, attr):
        raise NotImplementedError

    def conv_func(self, attr):
        raise NotImplementedError


@MUTATORS.register_module()
@CONV_LAYERS.register_module()
class Conv1dSlice(_ConvNdSlice):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 key=None,
                 channel_slice_mode='naive_order',
                 kernel_transform_mode='naive_order',
                 **kwargs):
        super(Conv1dSlice, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            0,
            groups,
            bias,
            padding_mode,
            key=key,
            channel_slice_mode=channel_slice_mode,
            kernel_transform_mode=kernel_transform_mode,
            **kwargs)

    def ntuple(self, attr):
        if isinstance(attr, (tuple, list)):
            return attr
        return (attr, )

    def conv_func(self, *args, **kwargs):
        return F.conv1d(*args, **kwargs)

    def export(self, **kwargs):
        """Export Conv1dSlice to nn.Conv1d."""
        (
            in_channels,
            out_channels,
            _,
            _,
            kernel_size,
            stride,
            padding,
            _,
            dilation,
            groups,
        ) = self.curr_attributes()
        in_channels = kwargs.get('in_channels', in_channels)
        out_channels = kwargs.get('out_channels', out_channels)
        kernel_size = kwargs.get('kernel_size', kernel_size)
        kernel_size = self.ntuple(kernel_size)
        stride = self.ntuple(kwargs.get('stride', stride))
        padding = self.ntuple(kwargs.get('padding', padding))
        dilation = self.ntuple(kwargs.get('dilation', dilation))
        groups = kwargs.get('groups', groups)

        export_module = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=(self.bias is not None))
        weight, bias = self.slice_params(in_channels, out_channels,
                                         kernel_size, groups)
        export_module.weight.data.copy_(weight.data)
        if self.bias is not None:
            export_module.bias.data.copy_(bias.data)

        return export_module


@MUTATORS.register_module()
@CONV_LAYERS.register_module()
class Conv2dSlice(_ConvNdSlice):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 key=None,
                 channel_slice_mode='naive_order',
                 kernel_transform_mode='naive_order',
                 **kwargs):
        super(Conv2dSlice, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            0,
            groups,
            bias,
            padding_mode,
            key=key,
            channel_slice_mode=channel_slice_mode,
            kernel_transform_mode=kernel_transform_mode,
            **kwargs)

    def ntuple(self, attr):
        if isinstance(attr, (tuple, list)):
            return attr
        return (attr, attr)

    def conv_func(self, *args, **kwargs):
        return F.conv2d(*args, **kwargs)

    def export(self, **kwargs):
        """Export Conv2dSlice to nn.Conv2d."""
        (
            in_channels,
            out_channels,
            _,
            _,
            kernel_size,
            stride,
            padding,
            _,
            dilation,
            groups,
        ) = self.curr_attributes()
        in_channels = kwargs.get('in_channels', in_channels)
        out_channels = kwargs.get('out_channels', out_channels)
        kernel_size = self.ntuple(kwargs.get('kernel_size', kernel_size))
        stride = self.ntuple(kwargs.get('stride', stride))
        padding = self.ntuple(kwargs.get('padding', padding))
        dilation = self.ntuple(kwargs.get('dilation', dilation))
        groups = kwargs.get('groups', groups)

        export_module = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=(self.bias is not None))
        weight, bias = self.slice_params(in_channels, out_channels,
                                         kernel_size, groups)
        export_module.weight.data.copy_(weight.data)
        if self.bias is not None:
            export_module.bias.data.copy_(bias.data)

        return export_module


@MUTATORS.register_module()
@CONV_LAYERS.register_module()
class Conv3dSlice(_ConvNdSlice):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 key=None,
                 channel_slice_mode='naive_order',
                 kernel_transform_mode='naive_order',
                 **kwargs):
        super(Conv3dSlice, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            0,
            groups,
            bias,
            padding_mode,
            key=key,
            channel_slice_mode=channel_slice_mode,
            kernel_transform_mode=kernel_transform_mode,
            **kwargs)

    def ntuple(self, attr):
        if isinstance(attr, (tuple, list)):
            return attr
        return (attr, attr, attr)

    def conv_func(self, *args, **kwargs):
        return F.conv3d(*args, **kwargs)

    def export(self, **kwargs):
        """Export Conv3dSearch to nn.Conv3d."""
        (
            in_channels,
            out_channels,
            _,
            _,
            kernel_size,
            stride,
            padding,
            _,
            dilation,
            groups,
        ) = self.curr_attributes()
        in_channels = kwargs.get('in_channels', in_channels)
        out_channels = kwargs.get('out_channels', out_channels)
        kernel_size = self.ntuple(kwargs.get('kernel_size', kernel_size))
        stride = self.ntuple(kwargs.get('stride', stride))
        padding = self.ntuple(kwargs.get('padding', padding))
        dilation = self.ntuple(kwargs.get('dilation', dilation))
        groups = kwargs.get('groups', groups)

        export_module = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=(self.bias is not None))
        weight, bias = self.slice_params(in_channels, out_channels,
                                         kernel_size, groups)
        export_module.weight.data.copy_(weight.data)
        if self.bias is not None:
            export_module.bias.data.copy_(bias.data)

        return export_module


@MUTATORS.register_module()
@CONV_LAYERS.register_module()
class ConvTranspose2dSlice(_ConvNdSlice):
    # TODO(sunyue): check realize.

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 output_padding=0,
                 key=None,
                 channel_slice_mode='naive_order',
                 kernel_transform_mode='naive_order',
                 **kwargs):
        super(ConvTranspose2dSlice, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            key=key,
            channel_slice_mode=channel_slice_mode,
            kernel_transform_mode=kernel_transform_mode,
            **kwargs)

    def forward_inner(self, x):
        # get current hyperparameters
        (
            in_channels,
            out_channels,
            in_group_channels,
            out_group_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        ) = self.curr_attributes()
        weight, bias = self.slice_params(in_channels, out_channels,
                                         kernel_size, groups)
        return self.conv_func(x, weight, bias, stride, padding, output_padding,
                              groups, dilation)

    def ntuple(self, attr):
        if isinstance(attr, (tuple, list)):
            return attr
        return (attr, attr)

    def conv_func(self, *args, **kwargs):
        return F.conv_transpose2d(*args, **kwargs)

    def export(self, **kwargs):
        """Export ConvTranspose2dSlice to nn.ConvTranspose2d."""
        (
            in_channels,
            out_channels,
            in_group_channels,
            out_group_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        ) = self.curr_attributes()
        in_channels = kwargs.get('in_channels', in_channels)
        out_channels = kwargs.get('out_channels', out_channels)
        kernel_size = self.ntuple(kwargs.get('kernel_size', kernel_size))
        stride = self.ntuple(kwargs.get('stride', stride))
        padding = self.ntuple(kwargs.get('padding', padding))
        output_padding = self.ntuple(
            kwargs.get('output_padding', output_padding))
        groups = kwargs.get('groups', groups)
        dilation = self.ntuple(kwargs.get('dilation', dilation))

        export_module = nn.ConvTranspose2d(
            out_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=(self.bias is not None),
            dilation=dilation)
        weight, bias = self.slice_params(in_channels, out_channels,
                                         kernel_size, groups)
        export_module.weight.data.copy_(weight.data)
        if self.bias is not None:
            export_module.bias.data.copy_(bias.data)

        return export_module
