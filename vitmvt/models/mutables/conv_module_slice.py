from mmcv.cnn import CONV_LAYERS, ConvModule, constant_init, kaiming_init

from ..builder import MUTATORS
from .base_mutable import SliceOp


@MUTATORS.register_module()
@CONV_LAYERS.register_module()  # TODO: clean and refactor
class ConvModuleSlice(ConvModule, SliceOp):
    """A conv block that bundles conv/norm/activation layers, which similar to
    mmcv.cnn.ConvModule.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer.
            Default: dict(type='Conv2dSlice'), which means using Conv2dSlice.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=dict(type='Conv2dSlice'),
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 key=None,
                 order=('conv', 'norm', 'act')):
        SliceOp.__init__(self, key=key)
        super(ConvModuleSlice, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=inplace,
            order=order)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners, and we do not want ConvModule to
        #    overrides the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners, they will be initialized by this method with default
        #    `kaiming_init`.
        # 3. For PyTorch's conv layers, they will be initialized anyway by
        #    their own `reset_parameters` methods.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x

    def export(self, **kwargs):
        # specify difference
        padding =\
            self.get_value(self.padding)if self.padding is not None else None
        export_module = ConvModule(
            kwargs.get('in_channels', self.get_value(self.in_channels)),
            kwargs.get('out_channels', self.get_value(self.out_channels)),
            kwargs.get('kernel_size', self.get_value(self.kernel_size)),
            stride=kwargs.get('stride', self.get_value(self.stride)),
            padding=kwargs.get('padding', padding),
            dilation=kwargs.get('dilation', self.get_value(self.dilation)),
            groups=kwargs.get('groups', self.get_value(self.groups)),
            inplace=self.inplace,
            order=self.order)
        export_module.with_norm = self.with_norm
        export_module.with_activation = self.with_activation
        export_module.with_bias = self.with_norm
        if self.with_norm:
            export_module.norm_name = self.norm_name
        for name in self._modules:
            layer = self.__getattr__(name)
            if isinstance(layer, SliceOp):
                export_module.__setattr__(name, layer.export())
            else:
                export_module.__setattr__(name, layer)

        return export_module
