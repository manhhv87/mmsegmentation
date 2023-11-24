def dcswin_tiny(pretrained=True, num_classes=4, weight_path='pretrain_weights/stseg_tiny.pth'):
    model = DCSwin(encoder_channels=(96, 192, 384, 768),
                   num_classes=num_classes,
                   embed_dim=96,
                   depths=(2, 2, 6, 2),
                   num_heads=(3, 6, 12, 24),
                   frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model