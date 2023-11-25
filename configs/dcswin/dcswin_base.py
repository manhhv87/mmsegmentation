_base_ = [
    '../_base_/models/dcswin.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint='https://drive.usercontent.google.com/download?id=1tHNxQUffwNIfWFDKa4ql1klKGjCnbhwv&export=download&authuser=0&confirm=t&uuid=3f0fc36c-9936-4d25-9187-33b43954afe5&at=APZUnTXkSlUg7GRvllIndBNc0acn:1700898173032'

model = dict(
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='DCSwin',
        encoder_channels=(128, 256, 512, 1024),
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        frozen_stages=2,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
    ),
    
    decode_head=dict(
        type='myFCNHead',
        in_channels=128,
        channels=128,
        num_classes=10,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.3),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.7)])
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                    'relative_position_bias_table': dict(decay_mult=0.),
                                    'norm': dict(decay_mult=0.)}))

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
