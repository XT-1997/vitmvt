from mmpose.models import TopDown

from ...builder import ARCHITECTURES


@ARCHITECTURES.register_module()
class TopDownSearch(TopDown):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None,
                 connect_head=None):
        super(TopDownSearch, self).__init__(
            backbone=backbone,
            neck=neck,
            keypoint_head=keypoint_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            loss_pose=loss_pose)

        if connect_head is not None:
            for kh, vh in connect_head.items():
                component, attr = vh.split('.')
                value = getattr(getattr(self, component), attr)
                from gml.models import Conv2dSlice
                self.neck = Conv2dSlice(value, keypoint_head['in_channels'], kernel_size=1)
