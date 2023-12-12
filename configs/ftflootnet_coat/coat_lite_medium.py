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
        type='CoaT',        
        patch_size=4, 
        embed_dims=[128, 256, 320, 512], 
        serial_depths=[3, 6, 10, 8], 
        parallel_depth=0, 
        num_heads=8, 
        mlp_ratios=[4, 4, 4, 4],
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
