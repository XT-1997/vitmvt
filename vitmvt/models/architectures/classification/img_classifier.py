try:
    from mmcls.models import ImageClassifier
except ImportError:
    from ....utils import get_placeholder
    ImageClassifier = get_placeholder('mmcls')

from ...builder import ARCHITECTURES, build_op


@ARCHITECTURES.register_module()
class ImageClassifierSearch(ImageClassifier):
    """ImageClassifierSearch for sliceable networks.
    Args:
        backbone (dict): The same as ImageClassifier.
        neck (dict, optional): The same as ImageClassifier. Defaults to None.
        head (dict, optional): The same as ImageClassifier. Defaults to None.
        pretrained (dict, optional): The same as ImageClassifier. Defaults to
            None.
        train_cfg (dict, optional): The same as ImageClassifier. Defaults to
            None.
        init_cfg (dict, optional): The same as ImageClassifier. Defaults to
            None.
        resize_process (dict, optional): Resize Process for the input data.
            Defaults to None.
        connect_head (dict, optional): keys in head will be substitute to it's
            `strtype` value, so that search_space of the first components can
            be connets to the next. e.g:
            {'in_channels': 'backbone.out_channels'} Means that heads'
            in_channels will be substitute to backbones out_channels. and
            rebuild the head component. Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None,
                 resize_process=None,
                 connect_head=None):
        # backbone.pretrained = '/mnt/cache/xietao/final_subnet_20220715_0843.pth'
        # backbone.pretrained = '/mnt/lustre/xietao/log2/1010/3x_sandwich4_seq_task_autoformer_cls/iter_150000.pth'
        # backbone.pretrained = '/mnt/lustre/xietao/LOG_NEW/debug_search/final_subnet_step40_20220902_0357.pth'
        # backbone.pretrained = '/mnt/lustre/xietao/ViT-MVT/base/best_accuracy_top-1_epoch_413.pth'
        # backbone.pretrained = '/mnt/lustre/xietao/log_mvt/tiny_search/final_subnet_step7_20221210_0316.pth'
        # backbone.pretrained = '/mnt/lustre/xietao/log_mvt/small_search/final_subnet_step25_20221212_0523.pth'
        # backbone.pretrained = '/mnt/lustre/xietao/ViT-MVT/tiny/best_accuracy_top-1_epoch_494.pth' 
        # backbone.pretrained = '/mnt/lustre/xietao/LOG_NEW/main_results_supernet/best_accuracy_top-1_epoch_442.pth'
        # backbone.pretrained = '/mnt/cache/xietao/best_accuracy_top-1_epoch_378.pth'
        # backbone.pretrained = '/mnt/lustre/xietao/LOG_NEW/debug_search/final_subnet_step55_20220902_0357.pth'
        # backbone.pretrained = '/mnt/lustre/xietao/log_mvt/base_search/final_subnet_step55_20221215_2011.pth'
        # backbone.pretrained = '/mnt/lustre/xietao/log2/main_results_mask_sharing/3x_task_autoformer_cls/iter_150000.pth'
        # backbone.pretrained = '/mnt/lustre/xietao/autoformer_series/cvpr_reb/cub_finetune_all_mask/best_accuracy_top-1_epoch_66.pth'
        super().__init__(
            backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            train_cfg=train_cfg,
            init_cfg=init_cfg)
        if resize_process is not None:
            self.resize_process = build_op(resize_process)
        if connect_head is not None:
            from mmcls.models import build_head
            for kh, vh in connect_head.items():
                component, attr = vh.split('.')
                value = getattr(getattr(self, component), attr)
                head[kh] = value
            self.head = build_head(head)

        # pretrained = '/mnt/cache/xietao/autorformer/supernet_no_clstoken/best_accuracy_top-1_epoch_378.pth'
        # if pretrained is not None:
        #     assert backbone.get('pretrained') is None, \
        #         'both backbone and classifier set pretrained weight'
        #     self.backbone.pretrained = pretrained

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if hasattr(self, 'resize_process'):
            img = self.resize_process(img)

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)
        drop_ratio = kwargs.pop('drop_ratio', 0.0)
        drop_path_ratio = kwargs.pop('drop_path_ratio', 0.0)
        x = self.extract_feat(
            img, drop_ratio=drop_ratio, drop_path_ratio=drop_path_ratio)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)
        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        if hasattr(self, 'resize_process'):
            img = self.resize_process(img)

        x = self.extract_feat(img)
        return self.head.simple_test(x)

    def forward_dummy(self, img, *args, **kwargs):
        """Used for computing network FLOPs.

        See ``evaluators->NaiveEvaluator``.
        """
        if hasattr(self, 'resize_process'):
            img = self.resize_process(img)
        x = self.extract_feat(img)
        return self.head.simple_test(x)

    def forward_pre_GAP(self,
                        img,
                        no_reslink=True,
                        drop_ratio=0,
                        drop_path_ratio=0):
        """Directly measure model-complexity from the backbone.

        no_reslink: without residual link in blocks.
        """
        x = self.backbone.forward_pre_GAP(
            img,
            no_reslink=no_reslink,
            drop_ratio=drop_ratio,
            drop_path_ratio=drop_path_ratio)
        return x

    def extract_feat(self, img, drop_ratio=0, drop_path_ratio=0):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(
            img, drop_ratio=drop_ratio, drop_path_ratio=drop_path_ratio)
        if self.with_neck:
            x = self.neck(x)
        return x
