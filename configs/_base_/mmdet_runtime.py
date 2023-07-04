checkpoint_config = dict(interval=1, max_keep_ckpts=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='PaviLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl', port=13456)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
default_scope = 'mmdet'
