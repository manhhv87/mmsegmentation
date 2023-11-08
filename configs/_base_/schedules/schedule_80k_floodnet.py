# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False)
]

# The `val_interval` is the original `evaluation.interval`.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')      # Use the default validation loop.
test_cfg = dict(type='TestLoop')    # Use the default test loop.

default_hooks = dict(
    # update runtime information, e.g. current iter and lr.
    runtime_info=dict(type='RuntimeInfoHook'),

    # record the time of every iterations.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', log_metric_by_epoch=True),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch, and automatically save the best checkpoint.
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1),

    # set sampler seed in distributed environment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # synchronize model buffers at the end of each epoch.
    sync_buffers=dict(type='SyncBuffersHook'),

    visualization=dict(type='SegVisualizationHook'))
