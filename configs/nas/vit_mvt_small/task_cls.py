_base_ = [
    '../../../configs/_base_/datasets/mmcls/imagenet_bs64_autoformer_224.py',
    '../../../configs/_base_/schedules/mmcls/imagenet_bs1024_AdamW.py',
    '../../../configs/_base_/mmcls_runtime.py'
]

custom_imports = dict(imports=[
    'mmcls.models', 'vitmvt.models'
])

model = dict(
    type='vitmvt.ImageClassifierSearch',
    backbone=dict(
        type='vitmvt.VIT_MVT_BACKBONE',
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
                data=[5, 6, 7],
                default=7,
                key='num_heads.0'),
            embed_channels=dict(
                type='Categorical',
                data=[320, 384, 448],
                default=448,
                key='embed_channels.0'))),
    neck=None,
    head=dict(
        type='vitmvt.LinearClsHead',
        num_classes=1000,
        in_channels=640,
        linear_type='DynamicLinearSlice',
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            mode='original',
            loss_weight=1.0,
            label_smooth_val=0.1,
            num_classes=1000),
        topk=(1, 5),
    ),
    connect_head=dict(in_channels='backbone.embed_dims'),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))

mutator = dict(type='vitmvt.StateslessMutator')
# custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]
algorithm = dict(
    type='vitmvt.ViT_MVT',
    model=model,
    retraining=True,
    mutator=mutator,
    grad_clip=dict(clip_type='by_value', clip_value=1.0),
    mutable_cfg='configs/nas/vit_mvt_small/vit_mvt_small.yaml')
find_unused_parameters = True


# TO DO：373 * 50
epoch_size = 10000
max_iters = 30 * epoch_size

test_setting = dict(
    repo='mmcls',  # call which repo's test function
    single_gpu_test=dict(show=False),
    multi_gpu_test=dict())

evaluation = dict(
    _delete_=True,
    type='mme.MultiTaskEvalHook',
    dataset={{_base_.data.val}},
    dataloader=dict(samples_per_gpu=128, workers_per_gpu=2),
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