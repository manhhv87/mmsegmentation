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
        type='SwinUnet',
        embed_dim=64,
        patch_size=4,
        depth=[1, 2, 21, 1],
        num_heads=[2, 4, 8, 16],
        split_size=[1, 2, 7, 7],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        init_cfg=dict(
          type='Pretrained', 
          checkpoint='https://objects.githubusercontent.com/github-production-release-asset-2e65be/382210636/87c0205e-d6c5-4055-9ce1-adb096034161?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231117%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231117T082017Z&X-Amz-Expires=300&X-Amz-Signature=b40a5bdf71fbca20a11ed7a26a3f71d4777e4a75d6faa7c460122ea2ea1a8d1d&X-Amz-SignedHeaders=host&actor_id=67886698&key_id=0&repo_id=382210636&response-content-disposition=attachment%3B%20filename%3Dcswin_base_224.pth&response-content-type=application%2Foctet-stream')),

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
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
