# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class LinearHead(BaseDecodeHead):
    def __init__(self, 
                 in_channels,
                 num_classes,                 
                 **kwargs):
        super(LinearHead, self).__init__(**kwargs)                
        self.fc = nn.Linear(in_channels, num_classes)    

    def forward(self, inputs):
        """Forward function."""
        output = self.fc(inputs)        
        return output
