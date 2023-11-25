# model settings
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
    backbone=dict(
        type='DCSwin',
        encoder_channels=(96, 192, 384, 768),
        dropout=0.05,
        atrous_rates=(6, 12),
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        frozen_stages=2,
        pretrained=None,
        init_cfg=None),

    decode_head=dict(
        type='myFCNHead',
        in_channels=96,
        in_index=0,
        channels=96,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=10,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
