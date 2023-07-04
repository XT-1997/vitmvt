# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='mmcv.TextLoggerHook')
    ])

# yapf:enable

dist_params = dict(backend='nccl', port=13245)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
default_scope = 'mmcls'
