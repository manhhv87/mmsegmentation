_base_ = [
    '../_base_/models/gss-ft-w_swin-l.py',
    '../_base_/datasets/floodnet.py',    
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
img_size = (768, 768)
model = dict(
    type='MultiDomainEncoderDecoder',
    decode_head=dict(
        type='GenerativeSegHeadFTW',
        num_classes=194,
        post_swin_depth=2,
        palette=[[2, 7, 214], [9, 16, 234], [3, 52, 5], [13, 65, 50],
                 [9, 52, 79], [5, 64, 106], [5, 67, 136], [3, 65, 180],
                 [9, 62, 215], [14, 59, 242], [6, 117, 8], [3, 104, 42],
                 [11, 115, 76], [2, 113, 115], [4, 114, 134], [5, 108, 167],
                 [2, 110, 203], [6, 118, 241], [14, 168, 13], [13, 156, 40],
                 [5, 158, 82], [9, 158, 101], [5, 155, 137], [9, 166, 169],
                 [14, 160, 212], [2, 165, 243], [12, 210, 13], [9, 215, 41],
                 [7, 214, 78], [12, 210, 108], [11, 208, 138], [0, 216, 177],
                 [0, 206, 200], [14, 210, 234], [58, 10, 9], [55, 15, 50],
                 [66, 7, 82], [70, 1, 114], [59, 4, 134], [63, 16, 174],
                 [68, 11, 205], [58, 16, 245], [61, 54, 6], [57, 52, 46],
                 [67, 64, 69], [60, 60, 104], [61, 62, 149], [58, 66, 167],
                 [65, 59, 203], [60, 67, 247], [66, 118, 14], [55, 117, 46],
                 [55, 109, 74], [67, 109, 113], [68, 108, 137], [64, 108, 181],
                 [65, 105, 213], [55, 116, 243], [69, 167, 9], [65, 162, 40],
                 [59, 166, 77], [60, 161, 101], [62, 158, 140], [63, 160, 178],
                 [70, 154, 207], [65, 158, 247], [61, 213, 4], [62, 215, 44],
                 [66, 218, 79], [57, 217, 108], [63, 212, 136], [60, 215, 179],
                 [55, 205, 200], [56, 210, 239], [112, 4, 14], [115, 6, 35],
                 [123, 5, 82], [119, 6, 109], [116, 5, 135], [121, 2, 181],
                 [111, 5, 212], [118, 6, 238], [116, 66, 5], [122, 53, 43],
                 [116, 59, 69], [110, 53, 112], [112, 55, 146], [117, 61, 175],
                 [120, 61, 211], [111, 57, 248], [124, 118, 11], [122, 106, 36],
                 [123, 105, 83], [110, 106, 108], [116, 115, 137],
                 [116, 104, 171], [111, 108, 209], [119, 105, 234],
                 [125, 167, 9], [122, 166, 48],
                 [119, 169, 72], [118, 163, 101], [115, 161, 137],
                 [116, 162, 167], [118, 162, 211], [123, 166, 236],
                 [124, 218, 2], [118, 209, 37], [124, 215, 83], [111, 211, 116],
                 [115, 219, 139], [124, 205, 176], [115, 219, 211],
                 [114, 207, 248], [171, 13, 15], [179, 10, 45], [172, 10, 79],
                 [171, 14, 116], [177, 6, 135], [173, 4, 182], [180, 1, 215],
                 [171, 8, 235], [171, 66, 5], [179, 57, 35], [168, 66, 70],
                 [171, 62, 107], [172, 56, 138], [169, 57, 178], [178, 67, 205],
                 [180, 67, 236], [178, 115, 10], [173, 117, 41], [179, 107, 79],
                 [177, 104, 115], [174, 104, 137], [166, 115, 169],
                 [167, 109, 212], [175, 116, 235], [172, 168, 15],
                 [176, 158, 36], [169, 159, 68], [165, 158, 115],
                 [167, 154, 147], [176, 154, 172], [165, 154, 206],
                 [171, 154, 247], [176, 216, 4], [166, 205, 50], [169, 216, 83],
                 [180, 216, 106], [176, 207, 140], [171, 205, 179],
                 [173, 219, 215], [180, 220, 246], [234, 3, 10], [229, 14, 46],
                 [224, 15, 69], [227, 7, 103], [230, 7, 148], [235, 13, 169],
                 [221, 2, 213], [229, 7, 235], [234, 62, 10], [235, 53, 38],
                 [226, 65, 68], [227, 60, 105], [229, 56, 145], [231, 54, 168],
                 [230, 52, 215], [232, 57, 240], [228, 103, 10], [221, 111, 47],
                 [229, 103, 70], [229, 118, 108], [234, 107, 138],
                 [230, 109, 177], [229, 113, 209], [234, 103, 239],
                 [231, 168, 17], [234, 154, 43], [227, 162, 83], [227, 164, 112],
                 [222, 156, 146], [233, 155, 174], [220, 158, 213],
                 [221, 163, 236], [228, 211, 4], [232, 220, 40], [233, 213, 78],
                 [233, 220, 113], [233, 210, 142], [223, 217, 173],
                 [225, 207, 207], [226, 220, 243], [0, 0, 0]]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))

optimizer = dict(
    type='AdamW',
    lr=0.0015,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=20000, metric='mIoU', pre_eval=True)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)