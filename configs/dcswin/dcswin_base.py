def dcswin_base(pretrained=True, num_classes=4, weight_path='pretrain_weights/stseg_base.pth'):
    # pretrained weights are load from official repo of Swin Transformer
    model = DCSwin(encoder_channels=(128, 256, 512, 1024),
                   num_classes=num_classes,
                   embed_dim=128,
                   depths=(2, 2, 18, 2),
                   num_heads=(4, 8, 16, 32),
                   frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model
