# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,

    backbone=dict(
        type='BANet',
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        apply_transform=True),

    decode_head=dict(
        type='GeneralHead',
        in_channels=256,
        channels=256,
        num_classes=10,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
