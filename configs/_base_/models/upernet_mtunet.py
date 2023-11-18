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
        type='MTUNet',
        num_heads=8,
        win_size=4,        
        bottleneck=1024,
        encoder=[256, 512],        
        decoder=[1024, 512]),

    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
        #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
         loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce',
                 use_sigmoid=False, loss_weight=0.3),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.7)]),

    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=384,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=19,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
