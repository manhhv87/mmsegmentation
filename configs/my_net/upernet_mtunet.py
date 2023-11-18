_base_ = [
    '../_base_/models/upernet_mtunet.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='MTUNet',
        num_heads=8,
        win_size=4,        
        bottleneck=1024,
        encoder=[256, 512],        
        decoder=[1024, 512]),

    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=10
    ),

    # auxiliary_head=dict(
    #     in_channels=384,
    #     num_classes=10)
    )

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=8000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=8000,
        end=80000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=8, num_workers=2)
val_dataloader = dict(batch_size=8, num_workers=2)
test_dataloader = val_dataloader
