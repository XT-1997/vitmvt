# yapf:disable
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='PaviLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl', port=31232)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
default_scope = 'mmseg'
