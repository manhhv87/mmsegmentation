_base_ = [
    '../_base_/models/upernet_mtunet.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

checkpoint='https://doc-0k-64-docs.googleusercontent.com/docs/securesc/tvoi3fua39rd37j1lobniftt2hb7dg0u/dtpc2nder1qr1k5fjntash0q1hfrj25e/1700288025000/15959628209787595289/09412009828125765653/1eo6d-d_kR0qbHBIHq49TQ1CFpPLypJUT?e=download&ax=AI0foUqvot2pbErllTdAGqMSMqekfgfS-9eJI7UbpVa9Qoek27EJZk1HvCjyHBO_ve4EWil-yjusjyHbQEe_dIcKg1XJ489EtlZfVxQo_-COj-stke_fGRsi6eJvdVZE95IKaDf79IHaaUm1XRJF1pgScoPiW9D5CGwLG2nA-ANUOkWIqdCLZXC0xYlclN_-1O_059kbO0yIAWVNWg0-ANuKGpS_6gEfir_lQm3vbFx0OZ6UnZpU3THPdoH9xF8t1YwKzHg9O7ZZwG0R0ogBsIZfRgfUkCyw0BZLXaPN6dNyeTtAOvd3DCnrVpNLRrdpWAYkVGX49m6dCo1cpSSiaCfAZwtKrBYqezHBSxT9Z7I-FuAvMet-MmDO-lQir88-jDL6KkPamY8S1EbQljzA4IpxAP0VmpJdXZky3fxU8d0KOJhPoyyiR9elkiwItsa1cG7eusUHrd8La9Fdl7RG8b9HFXx5kP0xiVTKdUJ07lJV_dAV618r8ayJY2t-BJ2ds49JPD8QUn1RwEdTanE9e1Z-fxcvr3uMF1ES78T56vzy0chPQovKtY1Eh4SrKnmp4Tg9qd0NH9yBEQQnkhnS_yoz6aVa7JhenV3RyGaLCKVeElZsJw6FEsCTJyU_68Z_XtgqlDNQXinuKdVUDBhK9f2C0nTJmVd4Po_KWXRVaWp_yta2z81QRdmL9VhfhIAwANFN27Oif9WZyoVErewMj9EQxddPic247vS_0rAe_1KH-WvoN82CikZRIJqqnv_MKVWG4ubUa7H6mKnpO3kX5I98Jeh2zx_7Mu2rCTSm03RCoEiKsqsToJ-HZ_v3lqVQUPoTZfo-OXbpOlAZbh2OYX4bXc5xDfEjAk-IYG8nnKTTAYGDFerM0TeW4B4pIxvIYpY3zmxW67tsk7J9rzR0DURr5WT5L54Iu1z2uB8G37fcBPD5pFO3BsHY4J5aC4oWSKb3mJbUaDKkQo1fjlq3XaosHlhJwPjtn_yuH7MjtSuP5dF4iqMWbkBRJM7khTOeVONPqMNsq02SdTJH9P5C&uuid=44bcb6cb-db01-4874-8cfb-01fedc8d059b&authuser=0&nonce=olg2pu1jlgsdc&user=09412009828125765653&hash=2ktmh88cffg7msfg9b5d3i95o8kd1a02'

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='MTUNet',
        num_heads=8,
        win_size=4,        
        bottleneck=1024,
        encoder=[256, 512],        
        decoder=[1024, 512],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),

    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=10
    ),

    # auxiliary_head=dict(
    #     in_channels=384,
    #     num_classes=10)
    )

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01))

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

train_dataloader = dict(batch_size=8, num_workers=2)
val_dataloader = dict(batch_size=8, num_workers=2)
test_dataloader = val_dataloader
