# Copyright (c) OpenMMLab. All rights reserved.
from .decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class ClsHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(ClsHead, self).__init__(**kwargs)

    def forward(self, inputs):
        """Forward function."""
        output = self.cls_seg(inputs)
        return output
