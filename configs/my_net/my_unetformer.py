_base_ = [
    '../_base_/models/my_unetformer.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k_floodnet.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=10))


optimizer=dict(type='AdamW', lr=0.0015, betas=(0.9, 0.999), weight_decay=0.3)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg = dict(
        norm_decay_mult=0.0, 
        bias_decay_mult=0.0),
    clip_gard=dict(max_norm=1.0))

param_scheduler = [
    # warmup
    dict(
        type='LinearLR', 
        start_factor=0.01, 
        by_epoch=True,         
        end=5,        
        convert_to_iter_based=True  # Update the learning rate after every iters.
        ),

    # main learning rate scheduler
    dict(type='CosineAnnealingLR', by_epoch=True, begin=5)
]

train_dataloader = dict(batch_size=8, num_workers=2)
val_dataloader = dict(batch_size=8, num_workers=2)
test_dataloader = val_dataloader
