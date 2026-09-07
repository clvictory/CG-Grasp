import torch
from torch import nn
from torch_scatter import scatter_mean
import torch.nn.functional as F


class ClusterContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1, intra_weight=1.0, inter_weight=1.0):
        super().__init__()
        self.temp = temp  # 温度参数控制相似度分布
        self.intra_weight = intra_weight  # 簇内紧凑性权重
        self.inter_weight = inter_weight  # 簇间分离性权重

    def forward(self, cluster_info):
        centers = cluster_info["value_centers"]  # [B, M, D]
        points = cluster_info["value_points"]  # [B, N, D]
        assignments = cluster_info["sim_max_idx"].squeeze(1)  # [B, N]

        # 分配中心到点
        batch_size, num_points, _ = points.shape
        expanded_assignments = assignments.unsqueeze(-1).expand(-1, -1, centers.size(-1))
        selected_centers = torch.gather(centers, 1, expanded_assignments)  # [B, N, D]

        # 簇内损失：最大化点与所属中心的相似度（即最小化 1 - sim）
        intra_sim = F.cosine_similarity(points, selected_centers, dim=-1) / self.temp  # 应用温度缩放
        intra_loss = (1 - intra_sim).mean() * self.intra_weight  # 越小表示簇内越紧凑

        # 簇间损失：最小化不同中心间的相似度
        centers_norm = F.normalize(centers, p=2, dim=-1)
        inter_sim = torch.bmm(centers_norm, centers_norm.transpose(1, 2))  # [B, M, M]
        mask = ~torch.eye(inter_sim.size(1), device=inter_sim.device).bool()  # 仅保留非对角线元素
        inter_sim = inter_sim.masked_select(mask.unsqueeze(0))  # [B, M*(M-1)]
        inter_loss = inter_sim.mean() * self.inter_weight  # 越小表示簇间越分离

        # 总损失 = 簇内紧凑性损失 + 簇间分离性损失
        total_loss = intra_loss + inter_loss
        # total_loss = intra_loss
        # total_loss = inter_loss
        return total_loss