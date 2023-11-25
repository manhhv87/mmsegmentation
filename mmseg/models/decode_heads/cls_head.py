# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from .decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class ClsHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(ClsHead, self).__init__(**kwargs)
        # self.m = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, inputs):
        """Forward function."""
        output = self.cls_seg(inputs)
        # self.m(output)
        return output
