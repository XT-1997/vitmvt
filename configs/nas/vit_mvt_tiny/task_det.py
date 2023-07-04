_base_ = [
    '../../../configs/_base_/datasets/mmdet/coco.py',
    '../../../configs/_base_/mmdet_runtime.py'
]

custom_imports = dict(imports=[
    'mmdet.models', 'vitmvt.models'
])

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (1024, 1024)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'mmdet.CocoDataset'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=1,
    train=dict(
        _delete_=True,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
   )

norm_cfg = dict(type='SyncBatchNormSlice', requires_grad=True)
model = dict(
    type='vitmvt.MaskRCNNSearch',
    # pretrained = '/mnt/lustre/xietao/log_mvt/tiny_search/final_subnet_step7_20221210_0316.pth',
    backbone=dict(
        type='vitmvt.VIT_MVT_BACKBONE',
        img_size=1333,
        out_indices=(13, ),
        use_window_att=True,
        search_space=dict(
            depth=dict(
                type='Categorical',
                data=[14],
                default=14,
                key='depth.0'),
            mlp_ratio=dict(
                type='Categorical',
                data=[3, 3.5, 4],
                default=4,
                key='mlp_ratio.0'),
            num_heads=dict(
                type='Categorical',
                data=[3, 4],
                default=4,
                key='num_heads.0'),
            embed_channels=dict(
                type='Categorical',
                data=[192, 216, 240],
                default=240,
                key='embed_channels.0'))),
    connect_head=dict(in_channels='backbone.embed_dims'),
    neck=dict(
        type='vitmvt.SFP',
        in_channels=[576, 576, 576, 576],
        out_channels=256,
        conv_cfg=dict(type='Conv2dSlice'),
        norm_cfg=norm_cfg,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        num_convs=2,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

mutator = dict(type='vitmvt.StateslessMutator')
# custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]
algorithm = dict(
    type='vitmvt.ViT_MVT',
    retraining=True,
    model=model,
    mutator=mutator,
    grad_clip=dict(clip_type='by_value', clip_value=1.0),
    mutable_cfg='configs/nas/vit_mvt_tiny/vit_mvt_tiny.yaml')
find_unused_parameters = True


# TO DO：373 * 50
epoch_size = 10000
max_iters = 30 * epoch_size

test_setting = dict(
    repo='mmdet',  # call which repo's test function
    single_gpu_test=dict(show=False),
    multi_gpu_test=dict())

evaluation = dict(
    _delete_=True,
    type='mme.MultiTaskEvalHook',
    dataset={{_base_.data.val}},
    dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),
    test_setting=test_setting,
    by_epoch=False,
    interval=epoch_size)

# optimizer

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00008,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

optimizer_config = dict(grad_clip=None)
# learning policy

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=250,
    warmup_by_epoch=False)


runner = dict(_delete_=True, type='mme.MultiTaskIterBasedRunner', max_iters=max_iters)

checkpoint_config = dict(interval=epoch_size)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='mmcv.TextLoggerHook', by_epoch=False),
        dict(
            type='PaviLoggerHook',
            init_kwargs=dict(project='autoformer_new'))
    ])


dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', max_iters)]