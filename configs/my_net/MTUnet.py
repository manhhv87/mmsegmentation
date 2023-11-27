_base_ = [
    '../_base_/models/mtunet.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint='https://doc-14-64-docs.googleusercontent.com/docs/securesc/tvoi3fua39rd37j1lobniftt2hb7dg0u/irkdqpi02npaso160pflk4unj2snlbrt/1701100275000/15959628209787595289/09412009828125765653/1frQAK05UtiAO8rvKG9y5GXABaH70_-Hu?e=download&ax=AEqgLxm9aogiP4YwsRD_QmTclJ8tNOG2NhSiG3zOooRpzkELLlCiJltEixjRAvIxhT-apgyYfRtBraYw-95HPq2yRlMQSCX-1Y2oowCKXObg5cRKG0rhOobZ0e229BFUMJtMlsWLVB5LAgq8xhOJT5kuhvETnVcjyy3fgTZntf5bS2VsHKJ5O7O3zUf-VedeFNrVOtm8NxKOn8ndHvHgBvUrxrqS66DMC0tbFXKKHRAMn_owSlqR3w72wqUalA_MyqXbuS7NZLberk3eOUtmJtvlbo_ypZ7GYQVI0WMhIfL9GniEFYeW1ujnmYBqRiUfzm_S1HqhHmlPVG4v9Fwy_isJ3t1TFUygEOITXt9P3JqM4ZNvUL34f8Te38pWz_t5fgBWh3DGkSDZK0KhNndzPbAmiW64i6IxV52o_9Wy3B7PZUyM9ZZQBi5r-OyCHmY5ft8LeCA8A8Zl7tWq4-nK6bW11rH86lse1xBklYKVZHjJL9oWAt41dZrPDrnLl1bu6xkM5lFS1g3IIWJxp4i3fDViI4sfN2PfUFxIMwzuR3ldSZVapFyPKYLHHIqiyBdylR4oDk8xStfC53ic7Qm40QYLc6GJU5299DVBTxxd7LJSJjULdWTbiYSNKsKrHMt_zbL0RXxV5OD5Tm6AmMg56wGWkuivfc2u2UHkBkA9Koz2KmiDlZyWZQ7qkQ5nvQ3XyNMKGMvzoMxUd8jTsq86-VMZRsddgHm7xoiFBH3esaVuZdBVPNRkpH6ChmWynyFT8iSkPLK7bgDB2rPJBiFX35XDyt6lAr3CoPAuaM-hTMBVEO2VWs3hmjSjCtx0WFFZ96KfYvAuvP8-xsKGmYemArl2nTFAI1MTqKGQNqz-1eukZLjZfAkEY-q5gF7B7SDUZ8ajbVnQdfnTnnO11eUA0sKUf5tOmgTt9GDBwyZwgensenU-nETGxGeEzMGrfit3ZIoKnM1jL2ivyIk3PuerCMblFljDtlxxXkCu2vJ3JVnE6M-XaasXpDgSRhtea1_6OBU2ltcozuIGSrxpiZhMert6XZEAqy5q5Kk&uuid=186e1afe-d1a1-4ae7-9240-9b189ba67758&authuser=0&nonce=flul6l88gnd9k&user=09412009828125765653&hash=pi01pdrgplmesra376mhp94nuhmn4sfa'

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,

    backbone=dict(
        type='MTUNet',
        type='MTUNet',
        encoder=[256, 512],
        decoder=[1024, 512],
        bottleneck=1024,
        num_heads=8,
        win_size=4),

    decode_head=dict(
        type='ClsHead',
        in_channels=64,
        in_index=0,
        channels=64,
        num_classes=10,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.3),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.7)]),
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
