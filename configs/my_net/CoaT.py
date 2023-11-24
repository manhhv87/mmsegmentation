# # CoaT.
# @register_model
# def coat_tiny(**kwargs):
#     model = CoaT(patch_size=4, embed_dims=[152, 152, 152, 152], serial_depths=[2, 2, 2, 2], parallel_depth=6, num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
#     model.default_cfg = _cfg_coat()
#     return model

# @register_model
# def coat_mini(**kwargs):
#     model = CoaT(patch_size=4, embed_dims=[152, 216, 216, 216], serial_depths=[2, 2, 2, 2], parallel_depth=6, num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
#     model.default_cfg = _cfg_coat()
#     return model

# @register_model
# def coat_small(**kwargs):
#     model = CoaT(patch_size=4, embed_dims=[152, 320, 320, 320], serial_depths=[2, 2, 2, 2], parallel_depth=6, num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
#     model.default_cfg = _cfg_coat()
#     return model

# # CoaT-Lite.
# @register_model
# def coat_lite_tiny(**kwargs):
#     model = CoaT(patch_size=4, embed_dims=[64, 128, 256, 320], serial_depths=[2, 2, 2, 2], parallel_depth=0, num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)
#     model.default_cfg = _cfg_coat()
#     return model

# @register_model
# def coat_lite_mini(**kwargs):
#     model = CoaT(patch_size=4, embed_dims=[64, 128, 320, 512], serial_depths=[2, 2, 2, 2], parallel_depth=0, num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)
#     model.default_cfg = _cfg_coat()
#     return model

# @register_model
# def coat_lite_small(**kwargs):
#     model = CoaT(patch_size=4, embed_dims=[64, 128, 320, 512], serial_depths=[3, 4, 6, 3], parallel_depth=0, num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)
#     model.default_cfg = _cfg_coat()
#     return model

# @register_model
# def coat_lite_medium(**kwargs):
#     model = CoaT(patch_size=4, embed_dims=[128, 256, 320, 512], serial_depths=[3, 6, 10, 8], parallel_depth=0, num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
#     model.default_cfg = _cfg_coat()
#     return model

# ABCNet: Attentive bilateral contextual network for efficient semantic segmentation of Fine-Resolution remotely sensed imagery
_base_ = [
    '../_base_/models/banet.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth'

model = dict(
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='CoaT',
        in_channels=3,
        patch_size=4, 
        in_channels=3, 
        embed_dims=[128, 256, 320, 512],
        serial_depths=[3, 6, 10, 8], 
        parallel_depth=0,
        num_heads=8, 
        mlp_ratios=[4, 4, 4, 4], 
        init_cfg=None),

    decode_head=dict(
        type='LinearHead',
        in_channels=512,        
        num_classes=10,       
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.3),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.7)])
)

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

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
