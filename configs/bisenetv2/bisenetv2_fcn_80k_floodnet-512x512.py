_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth'

model = dict(data_preprocessor=data_preprocessor)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(_delete_=True, type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=10))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
