_base_ = [
    '../_base_/models/floodnet_coat_lite.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://vcl.ucsd.edu/coat/pretrained/coat_lite_small_8d362f48.pth'  # noqa

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='CoaT',
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        serial_depths=[3, 4, 6, 3],
        parallel_depth=0,
        num_heads=8,
        mlp_ratios=[8, 8, 4, 4],
        drop_path_rate=0.0,
        out_features=["x1_nocls", "x2_nocls", "x3_nocls", "x4_nocls"],
        return_interm_layers=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
    ),

    decode_head=dict(
        type='UnetfloodnetHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=64,
        num_classes=10,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce',
                 use_sigmoid=False, loss_weight=0.3),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.7)])
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=None
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
