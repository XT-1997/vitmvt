_base_ = [
    '../../../configs/_base_/datasets/mmseg/ade20k.py',
    '../../../configs/_base_/mmseg_runtime.py',
    '../../../configs/_base_/schedules/mmseg/schedule_160k.py'
]

custom_imports = dict(imports=[
    'mmseg.models', 'vitmvt.models'
])

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        # flip=True,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, min_size=512),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    samples_per_gpu=1)

# model settings
norm_cfg = dict(type='SyncBatchNormSlice', requires_grad=True)
model = dict(
    type='vitmvt.EncoderDecoderSearch',
    # pretrained = '/mnt/lustre/xietao/log_mvt/tiny_search/final_subnet_step7_20221210_0316.pth',
    backbone=dict(
        type='vitmvt.VIT_MVT_BACKBONE',
        img_size=512,
        out_indices=(2, 5, 9, 13),
        use_window_att=False,
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
        type='MultiLevelNeck',
        conv_cfg=dict(type='Conv2dSlice'),
        in_channels=[576, 576, 576, 576],
        out_channels=576,
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        conv_cfg=dict(type='Conv2dSlice'),
        in_channels=[576, 576, 576, 576],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=576,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        conv_cfg=dict(type='Conv2dSlice'),
        in_channels=576,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

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
    repo='mmseg',  # call which repo's test function
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

log_config = dict(
    interval=50,
    hooks=[
        dict(type='mmcv.TextLoggerHook', by_epoch=False),
        dict(
            type='PaviLoggerHook',
            init_kwargs=dict(project='autoformer_new'))
    ])


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


dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', max_iters)]