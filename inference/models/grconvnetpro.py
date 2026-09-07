import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from kmeans_pytorch import kmeans
from inference.models.grasp_model import GraspModel, ResidualBlock


class GenerativeResnet(GraspModel):
    
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.1,
                 # 新增聚类相关参数
                 cluster_memory_size=1000, num_clusters=50, top_k=20,
                 prototype_threshold=0.1, 
                 # 多层级融合参数
                 use_multi_level_fusion=True,  # 是否使用多层级融合
                 fusion_levels=[1, 2, 3],  # 融合层级: 0=conv2, 1=conv3, 2=res5, 3=解码前
                 fusion_strength=0.5,  # 融合强度
                 adaptive_fusion=True,  # 自适应融合
                 prototype_dim=128,  # 原型向量维度
                 kmeans_device='gpu' if torch.cuda.is_available() else 'cpu'):
        
        super(GenerativeResnet, self).__init__()
        
        # 聚类模块参数
        self.cluster_memory_size = cluster_memory_size
        self.num_clusters = num_clusters
        self.top_k = top_k
        self.prototype_threshold = prototype_threshold
        self.prototype_dim = prototype_dim
        self.kmeans_device = kmeans_device
        
        # 多层级融合参数
        self.use_multi_level_fusion = use_multi_level_fusion
        self.fusion_levels = fusion_levels
        self.fusion_strength = fusion_strength
        self.adaptive_fusion = adaptive_fusion
        
        # 原型记忆库
        self.prototype_memory = {}
        self.cluster_centers = {}
        self.update_counter = {}
        
        # =================== 原始网络结构 ===================
        # 编码器
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)
        
        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)
        
        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)
        
        # 残差块
        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)
        
        # =================== 多层级原型提取模块 ===================
        if self.use_multi_level_fusion:
            # 定义4个关键层级：conv2输出, conv3输出, res5输出, 解码前
            self.feature_levels = 4
            level_dims = [
                channel_size * 2,  # conv2输出维度
                channel_size * 4,  # conv3输出维度
                channel_size * 4,  # res5输出维度
                channel_size * 4   # 解码前特征维度
            ]
            
            # 为每个层级创建投影头
            self.projection_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(level_dims[i], level_dims[i] // 2, kernel_size=1),
                    nn.BatchNorm2d(level_dims[i] // 2),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1)
                ) for i in range(self.feature_levels)
            ])
            
            # 为每个可融合层级创建原型感知卷积
            self.prototype_aware_convs = nn.ModuleList()
            for i, level_dim in enumerate(level_dims):
                if i in self.fusion_levels:
                    # 拼接原始特征和上下文特征
                    self.prototype_aware_convs.append(
                        nn.Sequential(
                            nn.Conv2d(level_dim + level_dim // 2, level_dim, kernel_size=3, padding=1),
                            nn.BatchNorm2d(level_dim),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    self.prototype_aware_convs.append(None)
            
            # 自适应融合权重
            if self.adaptive_fusion:
                self.fusion_weights = nn.ParameterList([
                    nn.Parameter(torch.ones(1) * 0.1)
                    for _ in range(len(self.fusion_levels))
                ])
        
        # =================== 解码器 ===================
        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)
        
        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)
        
        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)
        
        # 输出层
        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        
        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
    
    def extract_instance_prototypes(self, features, attention_map=None, level=0):
        """
        提取指定层级的实例原型
        
        Args:
            features: 特征图 [B, C, H, W]
            attention_map: 注意力图 [B, 1, H, W] (可选)
            level: 特征层级索引
            
        Returns:
            prototypes: 实例原型 [B, D]
        """
        B, C, H, W = features.shape
        
        # 如果未提供注意力图，使用特征图范数
        if attention_map is None:
            attention_map = torch.norm(features, dim=1, keepdim=True)
        elif attention_map.shape[2] != H or attention_map.shape[3] != W:
            # 调整注意力图尺寸
            attention_map = F.interpolate(
                attention_map,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        # 生成二值掩码
        binary_mask = (attention_map > self.prototype_threshold).float()
        
        # 使用对应层级的投影头
        if hasattr(self, 'projection_heads') and level < len(self.projection_heads):
            projection_head = self.projection_heads[level]
            # 投影到低维空间
            projected_features = projection_head(features)  # [B, D, 1, 1]
            projected_features = projected_features.squeeze(-1).squeeze(-1)  # [B, D]
            
            # 提取原型
            prototypes = []
            for i in range(B):
                if binary_mask[i].sum() > 0:
                    # 掩码平均池化
                    masked_features = features[i] * binary_mask[i]
                    prototype = masked_features.sum(dim=(1, 2)) / (binary_mask[i].sum() + 1e-8)
                    # 截断到投影维度
                    prototype = prototype[:projected_features.size(1)]
                    prototypes.append(prototype)
                else:
                    # 全局平均池化
                    prototype = features[i].mean(dim=(1, 2))[:projected_features.size(1)]
                    prototypes.append(prototype)
        else:
            # 回退到全局平均池化
            prototypes = features.mean(dim=(2, 3))  # [B, C]
            if prototypes.size(1) > self.prototype_dim:
                # 如果维度太大，使用随机投影
                if not hasattr(self, 'random_projection'):
                    self.random_projection = nn.Linear(prototypes.size(1), self.prototype_dim, bias=False)
                prototypes = self.random_projection(prototypes)
        
        return torch.stack(prototypes) if isinstance(prototypes, list) else prototypes
    
    def update_cluster_memory(self, prototypes, class_id=0, level=0):
        """更新指定层级的聚类记忆库"""
        try:
            if not hasattr(self, 'cluster_memory_size'):
                raise AttributeError("cluster_memory_size属性未初始化")
            
            if not hasattr(self, 'prototype_memory'):
                self.prototype_memory = {}
            
            # 创建层级特定的记忆库键
            memory_key = f"{class_id}_level{level}"
            
            if memory_key not in self.prototype_memory:
                self.prototype_memory[memory_key] = deque(maxlen=self.cluster_memory_size)
                self.cluster_centers[memory_key] = None
                self.update_counter[memory_key] = 0
            
            # 更新记忆库
            for proto in prototypes:
                self.prototype_memory[memory_key].append(proto.detach().cpu().numpy())
            
            self.update_counter[memory_key] += 1
            
            # 定期更新聚类中心
            if self.update_counter[memory_key] % 50 == 0:
                self.update_cluster_centers(memory_key)
                
        except Exception as e:
            print(f"更新聚类记忆库时出错: {e}")
            raise
    
    def get_context_prototypes(self, current_prototypes, class_id=0, level=0):
        """获取指定层级的上下文原型"""
        memory_key = f"{class_id}_level{level}"
        
        if (memory_key not in self.cluster_centers or 
            self.cluster_centers[memory_key] is None or
            len(self.cluster_centers[memory_key]) == 0):
            return current_prototypes
        
        try:
            cluster_centers = self.cluster_centers[memory_key].to(current_prototypes.device)
            
            # 计算余弦相似度
            current_prototypes_norm = F.normalize(current_prototypes, p=2, dim=1)
            cluster_centers_norm = F.normalize(cluster_centers, p=2, dim=1)
            
            # 计算相似度矩阵
            similarities = torch.matmul(current_prototypes_norm, cluster_centers_norm.t())
            
            # 选择top-k个最相似的聚类中心
            topk_similarities, topk_indices = torch.topk(similarities,
                                                         min(self.top_k, len(cluster_centers)),
                                                         dim=1)
            
            # 计算权重
            positiveness_weights = F.softmax(topk_similarities, dim=1)
            
            # 加权聚合
            selected_centers = cluster_centers[topk_indices]
            weighted_prototypes = torch.sum(
                selected_centers * positiveness_weights.unsqueeze(2),
                dim=1
            )
            
            return weighted_prototypes
            
        except Exception as e:
            # 如果出错，返回原始原型
            return current_prototypes
    
    def apply_prototype_fusion(self, features, level_idx, class_id=0):
        """
        应用原型融合到指定层级特征
        
        Args:
            features: 特征图 [B, C, H, W]
            level_idx: 层级索引
            class_id: 类别ID
            
        Returns:
            fused_features: 融合后的特征
        """
        if not self.use_multi_level_fusion or level_idx not in self.fusion_levels:
            return features
        
        B, C, H, W = features.shape
        
        # 提取当前原型
        prototypes = self.extract_instance_prototypes(features, level=level_idx)
        
        # 训练时更新记忆库
        if self.training:
            self.update_cluster_memory(prototypes, class_id, level_idx)
        
        # 获取上下文原型
        context_prototypes = self.get_context_prototypes(prototypes, class_id, level_idx)
        
        if context_prototypes is not None and len(context_prototypes) > 0:
            # 计算自适应融合强度
            if self.adaptive_fusion:
                # 随着层级的深入，融合强度增加
                fusion_idx = self.fusion_levels.index(level_idx)
                if fusion_idx < len(self.fusion_weights):
                    adaptive_strength = self.fusion_weights[fusion_idx].sigmoid()  # 限制在0-1之间
                else:
                    adaptive_strength = self.fusion_strength
            else:
                adaptive_strength = self.fusion_strength
            
            # 将原型广播到特征图尺寸
            context_features = context_prototypes.unsqueeze(-1).unsqueeze(-1)
            context_features = context_features.expand(B, -1, H, W)
            
            # 拼接并融合
            combined = torch.cat([features, context_features], dim=1)
            
            # 使用原型感知卷积
            conv_idx = level_idx if level_idx < len(self.prototype_aware_convs) else 0
            if self.prototype_aware_convs[conv_idx] is not None:
                fused_features = self.prototype_aware_convs[conv_idx](combined)
            else:
                # 简单的加权融合
                fused_features = features + adaptive_strength * context_features
            
            # 残差连接
            enhanced_features = features + adaptive_strength * fused_features
            return enhanced_features
        
        return features
    
    def update_cluster_centers(self, memory_key):
        """更新聚类中心"""
        if (memory_key not in self.prototype_memory or 
            len(self.prototype_memory[memory_key]) < max(2, self.num_clusters)):
            self.cluster_centers[memory_key] = None
            return
        
        try:
            prototypes = np.array(list(self.prototype_memory[memory_key]))
            prototypes_tensor = torch.tensor(prototypes, dtype=torch.float32)
            
            actual_clusters = min(self.num_clusters, len(prototypes))
            
            if actual_clusters < 2:
                self.cluster_centers[memory_key] = None
                return
            
            # 使用kmeans-pytorch
            cluster_ids_x, cluster_centers = kmeans(
                X=prototypes_tensor,
                num_clusters=actual_clusters,
                distance='euclidean',
                device=self.kmeans_device,
                tqdm_flag=False,
                seed=42
            )
            
            self.cluster_centers[memory_key] = cluster_centers
            
        except Exception as e:
            # 聚类失败，保持原样
            self.cluster_centers[memory_key] = None
    
    def forward(self, x_in, get_cluster=False, class_id=0):
        """
        前向传播 - 支持多层级原型融合
        
        Args:
            x_in: 输入图像
            get_cluster: 是否返回聚类信息
            class_id: 类别ID
            
        Returns:
            输出预测
        """
        # 存储中间特征用于原型融合
        intermediate_features = []
        
        # 编码器
        x = F.relu(self.bn1(self.conv1(x_in)))
        
        # 层级1: conv2输出
        x = F.relu(self.bn2(self.conv2(x)))
        if 0 in self.fusion_levels and self.use_multi_level_fusion:
            x = self.apply_prototype_fusion(x, level_idx=0, class_id=class_id)
        intermediate_features.append(x)
        
        # 层级2: conv3输出
        x = F.relu(self.bn3(self.conv3(x)))
        if 1 in self.fusion_levels and self.use_multi_level_fusion:
            x = self.apply_prototype_fusion(x, level_idx=1, class_id=class_id)
        intermediate_features.append(x)
        
        # 残差块
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        
        # 层级3: res5输出
        if 2 in self.fusion_levels and self.use_multi_level_fusion:
            x = self.apply_prototype_fusion(x, level_idx=2, class_id=class_id)
        intermediate_features.append(x)
        
        # 层级4: 解码前特征
        if 3 in self.fusion_levels and self.use_multi_level_fusion:
            x = self.apply_prototype_fusion(x, level_idx=3, class_id=class_id)
        
        # 解码器
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        
        # 输出层
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
        
        # 返回聚类信息（如果请求）
        if get_cluster:
            cluster_info = {
                'intermediate_features': intermediate_features,
                'prototype_memory_stats': self.get_cluster_statistics(class_id)
            }
            return pos_output, cos_output, sin_output, width_output, cluster_info
        
        return pos_output, cos_output, sin_output, width_output
    
    def get_cluster_statistics(self, class_id=0):
        """获取聚类统计信息"""
        stats = {}
        for level in self.fusion_levels:
            memory_key = f"{class_id}_level{level}"
            if memory_key in self.prototype_memory:
                memory_size = len(self.prototype_memory[memory_key])
                cluster_centers = 0 if self.cluster_centers[memory_key] is None else len(self.cluster_centers[memory_key])
                stats[f"level{level}"] = {
                    "memory_size": memory_size,
                    "cluster_centers": cluster_centers,
                    "update_counter": self.update_counter.get(memory_key, 0)
                }
        return stats
    
    def reset_memory(self, class_id=None):
        """重置记忆库"""
        if class_id is None:
            self.prototype_memory.clear()
            self.cluster_centers.clear()
            self.update_counter.clear()
        else:
            # 删除该类别所有层级的记忆
            keys_to_delete = []
            for key in list(self.prototype_memory.keys()):
                if key.startswith(f"{class_id}_"):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.prototype_memory[key]
                if key in self.cluster_centers:
                    del self.cluster_centers[key]
                if key in self.update_counter:
                    del self.update_counter[key]