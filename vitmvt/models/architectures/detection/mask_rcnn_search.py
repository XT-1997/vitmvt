from mmdet.models import MaskRCNN

from ...builder import ARCHITECTURES, build_op


@ARCHITECTURES.register_module()
class MaskRCNNSearch(MaskRCNN):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 connect_head=None):
        super(MaskRCNNSearch, self).__init__(
            backbone=backbone,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        if connect_head is not None:
            from mmdet.models import build_neck
            for kh, vh in connect_head.items():
                component, attr = vh.split('.')
                value = getattr(getattr(self, component), attr)
                neck[kh] = [value, value, value, value]
            self.neck = build_neck(neck)
