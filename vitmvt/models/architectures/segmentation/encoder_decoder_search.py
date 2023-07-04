from mmseg.models import EncoderDecoder

from ...builder import ARCHITECTURES


@ARCHITECTURES.register_module()
class EncoderDecoderSearch(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 connect_head=None):
        super().__init__(backbone, decode_head, neck, auxiliary_head,
                         train_cfg, test_cfg, pretrained, init_cfg)

        if connect_head is not None:
            from mmseg.models import builder
            for kh, vh in connect_head.items():
                component, attr = vh.split('.')
                value = getattr(getattr(self, component), attr)
                neck[kh] = [value, value, value, value]
                neck['out_channels'] = value
                decode_head[kh] = [value, value, value, value]
                auxiliary_head[kh] = value
            self.neck = builder.build_neck(neck)
            self._init_decode_head(decode_head)
            self._init_auxiliary_head(auxiliary_head)