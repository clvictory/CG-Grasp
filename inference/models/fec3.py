# Our model is build upon "Image as Set of Points", ICLR23. https://github.com/ma-xu/Context-Cluster/blob/main/models/context_cluster.py
# Thanks the authors for their impressive work!
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import time

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_
from timm.models import register_model
from timm.layers.helpers import to_2tuple
from einops import rearrange
import torch.nn.functional as F
from torch_scatter import scatter_sum
from inference.models.grasp_model import GraspModel




def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'model_small': _cfg(crop_pct=0.9),
    'model_medium': _cfg(crop_pct=0.95),
}


class PointReducer(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24):
        """
        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h

    def forward(self, x):  # [b,c,w,h]
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]计算相似度矩阵
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)# 找每个点最相似的中心
        mask = torch.zeros_like(sim)  # binary #[B,M,N] 初始化掩码
        mask.scatter_(1, sim_max_idx, 1.)# 硬分配（每个点仅分配到一个中心）
        sim = sim * mask# 应用掩码
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        M, N = value_centers.shape[1], value2.shape[1]
        value2 = rearrange(value2, 'b n c -> (b n) c')
        sim_max_idx = rearrange(sim_max_idx.squeeze(1), 'b n -> (b n)')
        idx_offset = (torch.arange(b, device=sim_max_idx.device) * M).unsqueeze(-1).expand(-1, N).flatten()
        sim_max_idx = sim_max_idx + idx_offset
        # 散列求和（将点特征聚合到中心）
        out = rearrange(scatter_sum(value2, sim_max_idx, dim=0, dim_size=b * M), '(b m) c -> b m c', b=b,
                        m=M)  # Different from CoC's implementation "(value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2)", we use scatter_sum to avoid OOM.
        out = (out + value_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)# 中心特征更新

        # dispatch step, return to each point in a cluster
        out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
        out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return out

    def get_cluster_info(self, x):
        # 单独的特征收集方法
        value = self.v(x)
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        M, N = value_centers.shape[1], value2.shape[1]
        value2 = rearrange(value2, 'b n c -> (b n) c')
        sim_max_idx = rearrange(sim_max_idx.squeeze(1), 'b n -> (b n)')
        idx_offset = (torch.arange(b, device=sim_max_idx.device) * M).unsqueeze(-1).expand(-1, N).flatten()
        sim_max_idx = sim_max_idx + idx_offset
        out = rearrange(scatter_sum(value2, sim_max_idx, dim=0, dim_size=b * M), '(b m) c -> b m c', b=b,
                        m=M)  # Different from CoC's implementation "(value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2)", we use scatter_sum to avoid OOM.
        out = (out + value_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)

        # dispatch step, return to each point in a cluster
        out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
        out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return {
            "sim": sim,
            "sim_max_idx": sim_max_idx,
            "value_centers": value_centers,
            "value_points": value2
        }


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,linear=False):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Cluster(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))

            # x = x

            # x = self.drop_path(
            #     self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
            #     * self.token_mixer(self.norm1(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(ClusterBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
            heads=heads, head_dim=head_dim
        ))
    blocks = nn.Sequential(*blocks)

    return blocks



class ConvDownsample(nn.Module):
    """
    优化后的卷积下采样模块
    改进点：
    1. 增加非线性激活函数
    2. 使用更合理的参数初始化
    3. 移除冗余的bias参数
    4. 提供灵活的下采样方式
    """

    def __init__(self, in_chans, out_chans, stride=2, act_layer=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans,
            out_chans,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False  # BN层已包含偏移参数，此处省略bias
        )
        self.bn = nn.BatchNorm2d(out_chans)
        self.act = act_layer(inplace=True)  # 使用通用激活函数接口

        # 参数初始化
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FEC(GraspModel):
    """
    feature extraction with clustering (FEC), the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, the embedding dims, mlp ratios
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained:
        for mmdetection and mmsegmentation to load pretrained weights
    """

    def __init__(self, layers, embed_dims=None, channel_size=64, dropout=True, prob=0.1,
                 mlp_ratios=None, downsamples=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=1000,
                 in_patch_size=4, in_stride=4, in_pad=1,
                 down_patch_size=2, down_stride=2, down_pad=0,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=True,
                 init_cfg=None,
                 pretrained=None,
                 proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2], fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1],
                 heads=[2, 4, 6, 8], head_dim=[16, 16, 32, 32], **kwargs):

        super().__init__()

        # 修改：统一的上采样模块，确保输出尺寸为224x224
        self.up_sample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dims[3], embed_dims[2], kernel_size=3, padding=1)
        )

        self.up_block1 = nn.Sequential(
            nn.Conv2d(embed_dims[2] * 2, embed_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims[2]),
            nn.ReLU(),
            nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims[2]),
            nn.ReLU()
        )

        self.up_sample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dims[2], embed_dims[1], kernel_size=3, padding=1)
        )

        self.up_block2 = nn.Sequential(
            nn.Conv2d(embed_dims[1] * 2, embed_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims[1]),
            nn.ReLU(),
            nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims[1]),
            nn.ReLU()
        )

        self.up_sample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dims[1], embed_dims[0], kernel_size=3, padding=1)
        )

        self.up_block3 = nn.Sequential(
            nn.Conv2d(embed_dims[0] * 2, embed_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU()
        )

        # 最终上采样到原尺寸
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dims[0], channel_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_size),
            nn.ReLU()
        )
        # ----------------------------------------------------------

        # 输出层
        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=1, kernel_size=1)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=1, kernel_size=1)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=1, kernel_size=1)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=1, kernel_size=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PointReducer(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=6, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 proposal_w=proposal_w[i], proposal_h=proposal_h[i],
                                 fold_w=fold_w[i], fold_h=fold_h[i], heads=heads[i], head_dim=head_dim[i],
                                 )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(ConvDownsample(
                in_chans=embed_dims[i],
                out_chans=embed_dims[i+1],
                stride=down_stride  # 通常设置为2
            ))

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    #中心裁剪函数（与UNet中的相同）
    def crop_tensor(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]
    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        _, c, img_w, img_h = x.shape
        # print(f"det img size is {img_w} * {img_h}")
        # register positional information buffer.
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)
        x = self.patch_embed(torch.cat([x, pos], dim=1))
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x,get_cluster=False):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # 获取不同层级的特征图
            s1 = x[0]  # (B, C, 56, 56)
            s2 = x[1]  # (B, C, 28, 28)
            s3 = x[2]  # (B, C, 14, 14)
            s4 = x[3]  # (B, C, 7, 7)

            # 新上采样路径 - 避免尺寸偏差
            # 1. 上采样s4到s3的尺寸
            d4 = self.up_sample1(s4)  # 7x7 -> 14x14 (双线性上采样确保尺寸精确)
            # 2. 拼接s3和d4 (注意: 两者尺寸现在都是14x14)
            d4 = torch.cat([s3, d4], dim=1)
            d4 = self.up_block1(d4)

            # 3. 上采样到s2的尺寸
            d3 = self.up_sample2(d4)  # 14x14 -> 28x28
            # 4. 拼接s2和d3 (两者都是28x28)
            d3 = torch.cat([s2, d3], dim=1)
            d3 = self.up_block2(d3)

            # 5. 上采样到s1的尺寸
            d2 = self.up_sample3(d3)  # 28x28 -> 56x56
            # 6. 拼接s1和d2 (两者都是56x56)
            d2 = torch.cat([s1, d2], dim=1)
            d2 = self.up_block3(d2)

            # 7. 最终上采样到224x224
            x = self.final_up(d2)
            if self.dropout:
                pos_output = self.pos_output(self.dropout_pos(x))
                cos_output = self.cos_output(self.dropout_cos(x))
                sin_output = self.sin_output(self.dropout_sin(x))
                width_output = self.width_output(self.dropout_wid(x))
            else:
                pos_output = self.pos_output(x)
                cos_output = self.cos_output(x)
                sin_output = self.sin_output(x)
                width_output = self.width_output(x)

            if get_cluster:  # 仅在需要时收集聚类信息
                cluster_info = []
                for block in self.network:
                    if isinstance(block, ClusterBlock):
                        cluster_info.append(block.token_mixer.get_cluster_info(x))
                return pos_output, cos_output, sin_output, width_output, cluster_info

            return pos_output, cos_output, sin_output, width_output
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out


@register_model
def fec_small(pretrained=False, **kwargs):
    layers = [3, 4, 5, 2]
    norm_layer = GroupNorm
    embed_dims = [32, 64, 196, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 2, 2]
    proposal_h = [4, 4, 2, 2]
    # proposal_w = [5, 5, 5, 5]
    # proposal_h = [5, 5, 5, 5]
    fold_w = [1, 1, 1, 1]
    fold_h = [1, 1, 1, 1]
    heads = [4, 4, 8, 8]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1
    model = FEC(
        layers, embed_dims=embed_dims, norm_layer=norm_layer, dropout=True, prob=0.1,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim, **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def fec_base(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 2, 2]
    proposal_h = [4, 4, 2, 2]
    fold_w = [1, 1, 1, 1]
    fold_h = [1, 1, 1, 1]
    heads = [4, 4, 8, 8]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = FEC(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim, **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def fec_large(pretrained=False, **kwargs):
    layers = [4, 4, 12, 4]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 2, 2]
    proposal_h = [4, 4, 2, 2]
    fold_w = [1, 1, 1, 1]
    fold_h = [1, 1, 1, 1]
    heads = [6, 6, 12, 12]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = FEC(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim, **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@torch.no_grad()
def compute_throughput(model, batch_size=256, resolution=224):
    import time
    torch.cuda.empty_cache()
    warmup_iters = 20
    num_iters = 100
    device = torch.device('cuda')

    model.eval()
    model.to(device)

    timing = []
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)

    # warmup
    for _ in range(warmup_iters):
        model(inputs)

    torch.cuda.synchronize()
    for _ in range(num_iters):
        start = time.time()
        model(inputs)
        torch.cuda.synchronize()
        timing.append(time.time() - start)

    timing = torch.as_tensor(timing, dtype=torch.float32)
    return (batch_size / timing.mean()).item()


def get_flops0():
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        net = fec_small()
        macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def get_flops1():
    from collections import Counter
    import numpy as np
    def fvcore_mul_flop_jit(inputs, outputs):
        flop_dict = Counter()
        flop_dict["mul"] = np.prod(inputs[0].type().sizes())
        return flop_dict

    input = torch.rand(1, 3, 224, 224)
    model = fec_small()
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, input)

    # flops.set_op_handle(**{'aten::mul': fvcore_mul_flop_jit, 'aten::div': fvcore_mul_flop_jit, 'aten::mul_': fvcore_mul_flop_jit, 'aten::add': fvcore_mul_flop_jit, 'aten::sum': fvcore_mul_flop_jit, 'aten::mean': fvcore_mul_flop_jit, 'aten::sub': fvcore_mul_flop_jit, 'aten::scatter_': fvcore_mul_flop_jit})

    print("FLOPs: ", flops.total() / 10. ** 9)


# if __name__ == '__main__':
#     input = torch.rand(8, 4, 224, 224)
#     model = fec_small()
#     pos_output, cos_output, sin_output, width_output = model(input)
#     # 创建SummaryWriter实例
#     # writer = SummaryWriter("logs/model_graph")  # 指定日志保存路径
#     #
#     # # 将模型和输入传递给writer
#     # writer.add_graph(model, input)
#     #
#     # # 关闭writer
#     # writer.close()

#     print(model)
#     print(pos_output, cos_output, sin_output, width_output)
#     # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     # print("number of params: {:.2f}M".format(n_parameters/1024**2))
#     #
#     # for i in range(3):
#     #     print(compute_throughput(model), end=' ')
#     #
#     # get_flops1()

def calculate_fps(model, input_shape, device='cuda', num_tests=100, warmup=10):
    """
    计算模型FPS
    :param model: 加载的模型
    :param input_shape: 输入张量维度 (batch, channel, height, width)
    :param device: 运行设备 ('cuda' 或 'cpu')
    :param num_tests: 正式测试次数
    :param warmup: 预热次数
    """
    # 准备输入数据
    dummy_input = torch.randn(input_shape).to(device)

    # 预热阶段 - 消除冷启动影响
    print("Warming up...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            if device == 'cuda':  # GPU同步
                torch.cuda.synchronize()

    # 正式计时测试
    print("Benchmarking...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_tests):
            model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    # 计算FPS
    fps = num_tests / total_time
    print(f"FPS: {fps:.2f} | Total Time: {total_time:.4f}s ({num_tests} runs)")
    return fps


# 使用示例
if __name__ == "__main__":
    # 1. 加载你的模型
    model = fec_small().eval()
    model = model.to('cuda:4')  # 或 'cpu'

    # 2. 计算FPS (batch=1, 3通道224x224输入)
    calculate_fps(model, input_shape=(1, 4, 224, 224), device='cuda:4')
