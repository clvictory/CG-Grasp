import torch
import torch.nn as nn
import torch.nn.functional as F
from inference.models.utils.TransUNet_Part import PatchMerging, PatchExpand, FinalPatchExpand_X4, BasicLayer, BasicLayer_up, PatchEmbed
from inference.models.utils.MobileUNet_Part import InvertedResidualBlock
from inference.models.utils.Bridge_Part import MtoT, TtoM
from inference.models.grasp_model import GraspModel
import math
from collections import deque
import numpy as np

# 尝试导入kmeans_pytorch，若不可用则提供一个简单的fallback（实际使用中建议安装）
from kmeans_pytorch import kmeans


class DSNetSys(GraspModel):
    def __init__(self, img_size=224, patch_size=4, in_chans=4, num_classes=1, embed_dim=96,
                 depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[1, 2, 4, 8],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first",
                 # ========== 新增聚类融合参数 ==========
                 cluster_memory_size=1000, num_clusters=50, top_k=20,
                 prototype_threshold=0.1, use_feature_alignment=True,
                 progressive_fusion=True, fusion_levels='high', fusion_strength=0.5,
                 adaptive_fusion=True, kmeans_device='gpu',
                 **kwargs):
        super().__init__()
        print("DSNet expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
            depths, depths_decoder, drop_path_rate, num_classes))

        # 原有参数
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # patch嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 编码器 layers (Swin)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # 解码器 layers_up (Swin)
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        # 最终上采样和输出
        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            # 输出通道数112 = embed_dim(96) + 16 (d5的通道数)
            self.pos_output = nn.Conv2d(in_channels=112, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.cos_output = nn.Conv2d(in_channels=112, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.sin_output = nn.Conv2d(in_channels=112, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.width_output = nn.Conv2d(in_channels=112, out_channels=self.num_classes, kernel_size=1, bias=False)

        # MobileUNet 部分
        self.conv3x3 = self.depthwise_conv(in_chans, 32, p=1, s=2)
        self.irb_bottleneck1 = self.irb_bottleneck(32, 16, 1, 1, 1)
        self.irb_bottleneck2 = self.irb_bottleneck(16, 24, 2, 2, 6)
        self.irb_bottleneck3 = self.irb_bottleneck(24, 32, 3, 2, 6)
        self.irb_bottleneck4 = self.irb_bottleneck(32, 96, 4, 2, 6)
        self.irb_bottleneck5 = self.irb_bottleneck(96, 1028, 3, 2, 6)
        # MobileUNet 解码部分
        self.D_irb1 = self.irb_bottleneck(1028, 96, 1, 2, 6, True)
        self.conv01 = nn.Conv2d(192, 96, 1)
        self.D_irb2 = self.irb_bottleneck(96, 32, 1, 2, 6, True)
        self.conv02 = nn.Conv2d(64, 32, 1)
        self.D_irb3 = self.irb_bottleneck(32, 24, 1, 2, 6, True)
        self.conv03 = nn.Conv2d(48, 24, 1)
        self.D_irb4 = self.irb_bottleneck(24, 16, 1, 2, 6, True)
        self.conv04 = nn.Conv2d(32, 16, 1)
        self.DConv4x4 = nn.ConvTranspose2d(16, 16, 4, 2, 1, groups=16, bias=False)

        # Bridge
        self.MtoT = MtoT()
        self.TtoM = TtoM()

        # ========== 新增：聚类融合相关模块 ==========
        # 聚类参数
        self.cluster_memory_size = cluster_memory_size
        self.num_clusters = num_clusters
        self.top_k = top_k
        self.prototype_threshold = prototype_threshold
        self.use_feature_alignment = use_feature_alignment
        self.kmeans_device = kmeans_device
        self.progressive_fusion = progressive_fusion
        self.fusion_levels = fusion_levels
        self.fusion_strength = fusion_strength
        self.adaptive_fusion = adaptive_fusion

        # 确定参与融合的层级索引 (0~3 对应 s1~s4)
        if self.fusion_levels == 's4_only':
            self.fusion_targets = [3]
        elif self.fusion_levels == 'high':
            self.fusion_targets = [2, 3]
        elif self.fusion_levels == 'all':
            self.fusion_targets = [0, 1, 2, 3]
        else:
            raise ValueError(f"不支持的fusion_levels: {self.fusion_levels}")

        # 四个层级的通道数 (与Swin编码器对应)
        self.enc_dims = [int(embed_dim * 2 ** i) for i in range(4)]  # [96, 192, 384, 768]
        self.proj_dims = [d // 2 for d in self.enc_dims]  # 投影维度

        # 多层级投影头
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.enc_dims[i], self.proj_dims[i], kernel_size=1),
                nn.BatchNorm2d(self.proj_dims[i]),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            ) for i in range(4)
        ])

        # 特征对齐偏移（可选）
        self.feature_shift = nn.Parameter(torch.zeros(self.proj_dims[3]))

        # 原型感知卷积（仅对目标层级创建）
        self.prototype_aware_convs = nn.ModuleList()
        for level in self.fusion_targets:
            in_ch = self.enc_dims[level] + self.proj_dims[level]
            out_ch = self.enc_dims[level]
            conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.prototype_aware_convs.append(conv)

        # 自适应融合权重（若需要）
        if self.adaptive_fusion:
            self.fusion_weights = nn.ParameterList([
                nn.Parameter(torch.ones(1) * 0.1) for _ in self.fusion_targets
            ])

        # 聚类记忆库
        self.prototype_memory = {}      # 存储原型
        self.cluster_centers = {}       # 存储聚类中心
        self.update_counter = {}         # 更新计数器

        # 权重初始化
        self.apply(self._init_weights)

    # ========== 原有辅助方法 ==========
    def depthwise_conv(self, in_c, out_c, k=3, s=1, p=0):
        conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=k, padding=p, groups=in_c, stride=s),
            nn.BatchNorm2d(num_features=in_c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=1),
        )
        return conv

    def irb_bottleneck(self, in_c, out_c, n, s, t, d=False):
        convs = []
        xx = InvertedResidualBlock(in_c, out_c, s, t, deconvolve=d)
        convs.append(xx)
        if n > 1:
            for i in range(1, n):
                xx = InvertedResidualBlock(out_c, out_c, 1, t, deconvolve=d)
                convs.append(xx)
        conv = nn.Sequential(*convs)
        return conv

    def get_count(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def up_x4(self, x, d5):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
            x = torch.cat((x, d5), dim=1)  # [B, 112, H, W]
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)
        return pos_output, cos_output, sin_output, width_output

    # ========== 新增：原型聚类融合相关方法 ==========
    def extract_instance_prototypes(self, features, attention_map, level):
        """
        features: [B, C, H, W]
        attention_map: [B, 1, H, W]  (或可广播)
        level: 层级索引 (0~3)
        """
        B, C, H, W = features.shape
        # 确保注意力图尺寸匹配
        if attention_map.shape[2] != H or attention_map.shape[3] != W:
            attention_map = F.interpolate(attention_map, size=(H, W), mode='bilinear', align_corners=False)

        binary_mask = (attention_map > self.prototype_threshold).float()
        proj_head = self.projection_heads[level]
        projected = proj_head(features)  # [B, D, 1, 1]
        projected = projected.squeeze(-1).squeeze(-1)  # [B, D]

        prototypes = []
        for i in range(B):
            if binary_mask[i].sum() > 0:
                masked = features[i] * binary_mask[i]
                proto = masked.sum(dim=(1,2)) / (binary_mask[i].sum() + 1e-8)
                proto = proto[:self.proj_dims[level]]  # 截断到投影维度
            else:
                proto = features[i].mean(dim=(1,2))[:self.proj_dims[level]]
            prototypes.append(proto)
        return torch.stack(prototypes)  # [B, D]

    def update_cluster_memory(self, prototypes, class_id=0, level=0):
        memory_key = f"{class_id}_level{level}"
        if memory_key not in self.prototype_memory:
            self.prototype_memory[memory_key] = deque(maxlen=self.cluster_memory_size)
            self.cluster_centers[memory_key] = None
            self.update_counter[memory_key] = 0

        for proto in prototypes:
            self.prototype_memory[memory_key].append(proto.detach().cpu().numpy())

        self.update_counter[memory_key] += 1
        if self.update_counter[memory_key] % 50 == 0:
            self.update_cluster_centers(memory_key)

    def get_context_prototypes(self, current_prototypes, class_id=0, level=0):
        memory_key = f"{class_id}_level{level}"
        if (memory_key not in self.cluster_centers or
            self.cluster_centers[memory_key] is None or
            len(self.cluster_centers[memory_key]) == 0):
            return current_prototypes

        cluster_centers = self.cluster_centers[memory_key].to(current_prototypes.device)
        # 余弦相似度
        cur_norm = F.normalize(current_prototypes, p=2, dim=1)
        center_norm = F.normalize(cluster_centers, p=2, dim=1)
        sim = torch.matmul(cur_norm, center_norm.t())
        topk_sim, topk_idx = torch.topk(sim, min(self.top_k, len(cluster_centers)), dim=1)
        weights = F.softmax(topk_sim, dim=1)
        selected = cluster_centers[topk_idx]
        weighted = torch.sum(selected * weights.unsqueeze(2), dim=1)
        return weighted

    def apply_progressive_fusion(self, features, proto_dict, class_id=0):
        """
        features: 列表 [s1, s2, s3, s4] (图像格式)
        proto_dict: 字典，每个层级对应的原型 [B, proj_dim]
        """
        enhanced = list(features)
        sorted_targets = sorted(self.fusion_targets, reverse=True)  # 从高层到低层
        for idx, level in enumerate(sorted_targets):
            feat = enhanced[level]
            B, C, H, W = feat.shape

            # 自适应融合强度
            if self.adaptive_fusion:
                strength = self.fusion_strength * (0.8 ** idx)
            else:
                strength = self.fusion_strength

            # 获取该层级的上下文原型（使用对应层级的原型检索）
            current_protos = proto_dict[level]
            context = self.get_context_prototypes(current_protos, class_id, level)
            if context is None or context.numel() == 0:
                continue

            # 广播到特征图尺寸
            context_feat = context.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)

            # 根据层级选择融合方式
            if level in self.fusion_targets and level >= 2:  # 高层使用原型感知卷积
                combined = torch.cat([feat, context_feat], dim=1)
                conv_idx = self.fusion_targets.index(level)
                if conv_idx < len(self.prototype_aware_convs):
                    fused = self.prototype_aware_convs[conv_idx](combined)
                else:
                    fused = feat + strength * context_feat
            else:  # 低层使用加权融合
                if self.adaptive_fusion and level in self.fusion_targets:
                    w_idx = self.fusion_targets.index(level)
                    if w_idx < len(self.fusion_weights):
                        weight = self.fusion_weights[w_idx]
                        fused = feat + weight * context_feat
                    else:
                        fused = feat + strength * context_feat
                else:
                    fused = feat + strength * context_feat

            enhanced[level] = feat + strength * fused  # 残差连接
        return enhanced

    def update_cluster_centers(self, memory_key):
        if (memory_key not in self.prototype_memory or
            len(self.prototype_memory[memory_key]) < max(2, self.num_clusters)):
            self.cluster_centers[memory_key] = None
            return

        try:
            prototypes = np.array(list(self.prototype_memory[memory_key]))
            proto_tensor = torch.tensor(prototypes, dtype=torch.float32)
            actual_clusters = min(self.num_clusters, len(prototypes))
            if actual_clusters < 2:
                self.cluster_centers[memory_key] = None
                return
            # 使用kmeans_pytorch
            _, cluster_centers = kmeans(
                X=proto_tensor,
                num_clusters=actual_clusters,
                distance='euclidean',
                device=self.kmeans_device,
                tqdm_flag=False,
                seed=42
            )
            self.cluster_centers[memory_key] = cluster_centers
        except Exception as e:
            self.cluster_centers[memory_key] = None

    def get_cluster_statistics(self, class_id=0):
        stats = {}
        for level in self.fusion_targets:
            key = f"{class_id}_level{level}"
            if key in self.prototype_memory:
                stats[f"level{level}"] = {
                    "memory_size": len(self.prototype_memory[key]),
                    "cluster_centers": 0 if self.cluster_centers[key] is None else len(self.cluster_centers[key]),
                    "update_counter": self.update_counter.get(key, 0)
                }
        return stats

    def reset_memory(self, class_id=None):
        if class_id is None:
            self.prototype_memory.clear()
            self.cluster_centers.clear()
            self.update_counter.clear()
        else:
            keys = [k for k in self.prototype_memory if k.startswith(f"{class_id}_")]
            for k in keys:
                del self.prototype_memory[k]
                if k in self.cluster_centers:
                    del self.cluster_centers[k]
                if k in self.update_counter:
                    del self.update_counter[k]

    # ========== 修改后的forward ==========
    def forward(self, x, class_id=0):
        # ---------- MobileUNet 编码 ----------
        x1 = self.conv3x3(x)                # [B,32,112,112]
        x2 = self.irb_bottleneck1(x1)       # [B,16,112,112]
        x3 = self.irb_bottleneck2(x2)       # [B,24,56,56]
        x4 = self.irb_bottleneck3(x3)       # [B,32,28,28]
        x5 = self.irb_bottleneck4(x4)       # [B,96,14,14]
        x6 = self.irb_bottleneck5(x5)       # [B,1028,7,7]

        # ---------- Swin Transformer 编码 ----------
        x_patch = self.patch_embed(x)        # [B, L, C]  L=H/4 * W/4
        if self.ape:
            x_patch = x_patch + self.absolute_pos_embed
        x_patch = self.MtoT(x6, x_patch)     # [B, L, C]

        # 收集每个 stage 的输入特征（跳跃连接），从高分辨率到低分辨率
        x_downsample_raw = []
        x_curr = x_patch
        for layer in self.layers:
            x_downsample_raw.append(x_curr)   # 保存进入当前 stage 前的特征
            x_curr = layer(x_curr)            # 经过当前 stage（可能下采样）
        x = x_curr                             # 最终编码输出（最低分辨率）

        # 将序列特征转换为图像格式，得到 s1~s4（从高分辨率到低分辨率）
        enc_imgs = []
        for feat_seq in x_downsample_raw:
            B, L, C = feat_seq.shape
            H = W = int(math.sqrt(L))
            feat_img = feat_seq.permute(0,2,1).reshape(B, C, H, W)
            enc_imgs.append(feat_img)
        s1, s2, s3, s4 = enc_imgs   # s1:96,56,56; s2:192,28,28; s3:384,14,14; s4:768,7,7

        # 生成注意力图（用于原型提取）
        attn_maps = [torch.norm(f, dim=1, keepdim=True) for f in [s1, s2, s3, s4]]

        # ========== 原型聚类融合 ==========
        if self.progressive_fusion:
            # 提取所有目标层级的原型
            proto_dict = {}
            for level in self.fusion_targets:
                feat = [s1, s2, s3, s4][level]
                attn = attn_maps[level]
                protos = self.extract_instance_prototypes(feat, attn, level=level)
                proto_dict[level] = protos
                if self.training:
                    self.update_cluster_memory(protos, class_id, level=level)

            # 渐进式融合
            enhanced = self.apply_progressive_fusion([s1, s2, s3, s4], proto_dict, class_id)
            s1_enh, s2_enh, s3_enh, s4_enh = enhanced
        else:
            # 单层融合（仅 s4）
            s4_protos = self.extract_instance_prototypes(s4, attn_maps[3], level=3)
            if self.training:
                self.update_cluster_memory(s4_protos, class_id, level=3)
            context = self.get_context_prototypes(s4_protos, class_id, level=3)
            if context is not None and context.numel() > 0:
                B, C, H, W = s4.shape
                context_feat = context.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
                combined = torch.cat([s4, context_feat], dim=1)
                if len(self.prototype_aware_convs) > 0:
                    s4_enh = self.prototype_aware_convs[0](combined)
                else:
                    s4_enh = s4
            else:
                s4_enh = s4
            s1_enh, s2_enh, s3_enh = s1, s2, s3

        # 将增强后的特征转换回序列格式，构建新的跳跃连接列表（顺序与 x_downsample_raw 一致：从高分辨率到低分辨率）
        def img_to_seq(feat_img):
            B, C, H, W = feat_img.shape
            return feat_img.reshape(B, C, H*W).permute(0,2,1)   # [B, L, C]

        enhanced_seqs = [img_to_seq(s1_enh), img_to_seq(s2_enh), img_to_seq(s3_enh), img_to_seq(s4_enh)]

        # ---------- MobileUNet 解码 ----------
        d1 = torch.cat((self.D_irb1(x6), x5), dim=1)
        d1 = self.conv01(d1)
        d2 = torch.cat((self.D_irb2(d1), x4), dim=1)
        d2 = self.conv02(d2)
        d3 = torch.cat((self.D_irb3(d2), x3), dim=1)
        d3 = self.conv03(d3)
        x_for_Mobile = self.TtoM(d3, x)      # x 是最终编码输出（序列格式）
        d4 = torch.cat((self.D_irb4(x_for_Mobile), x2), dim=1)
        d4 = self.conv04(d4)
        d5 = self.DConv4x4(d4)                # [B,16,224,224]

        # ---------- Swin Transformer 解码 ----------
        # 使用增强后的跳跃连接 enhanced_seqs（顺序与原代码 x_downsample 一致：从高到低）
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, enhanced_seqs[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)   # [B, L, C]

        # 最终上采样和输出
        pos, cos, sin, width = self.up_x4(x, d5)
        return pos, cos, sin, width