# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',

    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),

    decode_head=dict(
        type='UnetformerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=64,
        dropout_ratio=0.1,
        num_classes=10,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    
    # auxiliary_head=[
    #     dict(
    #         type='FCNHead',                 # Type of auxiliary head. Please refer to mmseg/models/decode_heads for available options.
    #         in_channels=16,                 # Input channel of auxiliary head.
    #         channels=16,                    # The intermediate channels of decode head.
    #         num_convs=2,                    # Number of convs in FCNHead. It is usually 1 in auxiliary head.
    #         num_classes=10,                 # Number of segmentation class.
    #         in_index=1,                     # The index of feature map to select.
    #         norm_cfg=norm_cfg,              # The configuration of norm layer.
    #         concat_input=False,             # Whether concat output of convs with input before classification layer.
    #         align_corners=False,            # The align_corners argument for resize in decoding.
    #         loss_decode=dict(               # Config of loss function for the auxiliary_head.
    #             type='CrossEntropyLoss',    # Type of loss used for segmentation.
    #             use_sigmoid=False,          # Whether use sigmoid activation for segmentation.
    #             loss_weight=1.0)),          # Loss weight of auxiliary_head.
    #     dict(
    #         type='FCNHead',
    #         in_channels=32,
    #         channels=64,
    #         num_convs=2,
    #         num_classes=10,
    #         in_index=2,
    #         norm_cfg=norm_cfg,
    #         concat_input=False,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    #     dict(
    #         type='FCNHead',
    #         in_channels=64,
    #         channels=256,
    #         num_convs=2,
    #         num_classes=10,
    #         in_index=3,
    #         norm_cfg=norm_cfg,
    #         concat_input=False,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    #     dict(
    #         type='FCNHead',
    #         in_channels=128,
    #         channels=1024,
    #         num_convs=2,
    #         num_classes=10,
    #         in_index=4,
    #         norm_cfg=norm_cfg,
    #         concat_input=False,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # ],
    
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
