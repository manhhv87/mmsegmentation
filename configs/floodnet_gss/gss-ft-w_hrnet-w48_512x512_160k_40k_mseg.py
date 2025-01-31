_base_ = [
    '../_base_/models/gss-ft-w_hrnet-w48.py',
    '../_base_/datasets/floodnet.py',    
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

optimizer = dict(
    type='AdamW',
    lr=0.00012,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)