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
    pretrained=None,

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
        ),

    decode_head=dict(
        type='UnetfloodnetHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=64,
        dropout_ratio=0.1,
        num_classes=10,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
