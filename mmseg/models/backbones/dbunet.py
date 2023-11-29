import warnings
import torch
import torch.nn as nn
# import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# from mmcv.cnn import ConvModule

from mmengine.model import BaseModule
from mmseg.registry import MODELS


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# Bock of Transformer Encoder
# 1. Fedd Forward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# 2. Attention
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(64, 12, bias=False),  # Data set for 224
            nn.ReLU(inplace=True),
            nn.Linear(12, 64, bias=False),
            nn.Sigmoid()
        )
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(1)
        self.PatchConv_stride1 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.PatchConv_stride1_bn = nn.BatchNorm2d(1)
        self.PatchConv_stride1_rl = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList([])
        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads=heads,
                    dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, x, tokens, x_AfterPatchEmbedding):
        # Create a new empty tensor to store the first 64 patches
        x_AfterConv = torch.zeros([1, 1, 64, 196])
        for i in range(64):  # Iterate over each row (each)
            # Get the value of this row
            x_patch_i = x_AfterPatchEmbedding[0][i][:]
            x_patch_i = x_patch_i.view(1, 1, 1, 196)
            x_patch_i = self.PatchConv_stride1(x_patch_i)
            x_patch_i = self.PatchConv_stride1_bn(x_patch_i)
            x_patch_i = self.PatchConv_stride1_rl(x_patch_i)
            # Assign value to the newly created tensor x_AfterConv.shape=[1, 1, 64, 196]
            x_AfterConv[0][0][i][:] = x_patch_i[0][0][0][:]
        a = x_AfterConv.view(1, 64, 1, 196)
        # a = a.to(DEVICE)
        # avgpool, compresses a tensor of size [1, 64]
        b1 = nn.AdaptiveAvgPool2d(1)(a).view(1, 64)
        # b1 = b1.to(DEVICE)

        val, idx = torch.sort(b1)  # Sort in ascending order
        # If all values are < 0, or all values are > 0,
        if torch.max(val) < 0 or torch.min(val) > 0:
            middle = 32
        else:
            for i in range(64):
                if val[0][i] > 0:
                    # Find the middle value (the intersection of negative and positive)
                    middle = i
                    break
                if i == 63:
                    middle = 32
        suppress = idx[0][:middle]
        excitation = idx[0][middle:]
        l_s = len(suppress)
        l_e = len(excitation)
        b1[0][suppress] = b1[0][suppress] - 1 / \
            (1 + l_s ** (b1[0][suppress])
             )  # Anti-sigmoid, the base is the length
        b1[0][excitation] = b1[0][excitation] + 1 / \
            (1 + l_e ** -(b1[0][excitation])
             )  # sigmoid, the base is the length
        b = b1

        # The previous (1, 64) is the size of b
        c = self.fc(b).view(1, 64, 1, 1)
        x_attention = a * c.expand_as(a)
        # [1, 64, 1, 196] -> [1, 64, 196]
        x_attention = x_attention.view(1, 64, 196)
        token = tokens.view(1, 1, 196)
        x_attention = torch.cat([x_attention, token], dim=1)  # [1, 65, 196]

        for attn, ff in self.layers:
            x = attn(x) + x + x_attention
            x = ff(x) + x
        return x


# FFB module
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        # Firstly, the channel dimension is increased by 1 * 1 convolution,
        # and the number of channels to be trained is fixed at the early stage of Bottleneck.
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, groups=4, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Then, 3 * 3 convolution is used to realize the first feature extraction.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, groups=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Next, 1 * 1 convolution is employed again to achieve feature information fusion with the same number of channels.
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, groups=4, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        # Finally, 3 * 3 and 1 * 1 convolutions are performed respectively to realize feature re-extraction and information re-fusion.
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, groups=2, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)

        self.conv5 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, groups=4, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Firstly, the channel dimension is increased by 1 * 1 convolution,
        # and the number of channels to be trained is fixed at the early stage of Bottleneck.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        residual1 = out

        # Then, 3 * 3 convolution is used to realize the first feature extraction.
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        residual2 = out

        # Next, 1 * 1 convolution is employed again to achieve feature information fusion with the same number of channels.
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = out + residual1

        # Finally, 3 * 3 convolution is performed to realize feature re-extraction.
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = out + residual2

        # Finally, 1 * 1 convolution is used to realize information re-fusion.
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TransEncoder(nn.Module):
    def __init__(self,
                 image_size=224,
                 in_channels=3,
                 patch_size=28,
                 dim=196,
                 heads=16,          # 224/28 + 224/28
                 mlp_dim=2048,
                 dim_head=64,       # 224/28 * 224/28
                 dropout=0.,
                 emb_dropout=0.,
                 **kwargs):
        super().__init__(**kwargs)

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x_AfterPatchEmbedding = x
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        vit_layerInfo = []
        for i in range(4):  # Where to set the depth [6, 64+1, dim=196]
            x = self.transformer(
                x, cls_tokens[i], x_AfterPatchEmbedding)
            vit_layerInfo.append(x)

        return vit_layerInfo


@MODELS.register_module()
class DBUNet(BaseModule):
    """Context Path to provide sufficient receptive field.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        context_channels (Tuple[int]): The number of channel numbers
            of various modules in Context Path.
            Default: (128, 256, 512).
        align_corners (bool, optional): The align_corners argument of
            resize operation. Default: False.
    Returns:
        x_16_up, x_32_up (torch.Tensor, torch.Tensor): Two feature maps
            undergoing upsampling from 1/16 and 1/32 downsampling
            feature maps. These two feature maps are used for Feature
            Fusion Module and Auxiliary Head.
    """

    def __init__(self,
                 backbone_cfg,
                 img_size=224,
                 in_channels=3,
                 patch_size=28,
                 dim=196,
                 depth=6,
                 heads=16,
                 mlp_dim=2048,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'

        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        self.backbone = MODELS.build(backbone_cfg)

        self.transencoder = TransEncoder(image_size=img_size, patch_size=patch_size, dim=dim,
                                         depth=depth, heads=heads, mlp_dim=mlp_dim, in_channels=in_channels,
                                         dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)

    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone(x)
        vit_layerInfo = self.transencoder(x)

        return tuple(x_4, x_8, x_16, x_32), vit_layerInfo
