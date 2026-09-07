# inference/pioir.py

import torch
import torch.nn as nn
from torchvision import models
from sklearn.cluster import KMeans
import numpy as np

class DeepFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用ResNet18提高效率
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 修改第一层以适应4通道输入
        original_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=4, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # 复制RGB权重到新增通道
        with torch.no_grad():
            self.model.conv1.weight[:, :3] = original_conv.weight.clone()
            self.model.conv1.weight[:, 3] = original_conv.weight.mean(dim=1)
        
        # 移除最后的池化层和全连接层
        self.feature_extractor = nn.Sequential(
            *list(self.model.children())[:-2]
        )
        
        # 冻结部分权重
        for name, param in self.feature_extractor.named_parameters():
            if "conv1" not in name:  # 只训练第一卷积层
                param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)

def extract_topk_features(image_tensor, k=5):
    """从输入图像中提取TOP-K聚类特征"""
    device = image_tensor.device
    batch_size = image_tensor.size(0)
    
    # 初始化特征提取器并移动到相同设备
    extractor = DeepFeatureExtractor().to(device).eval()
    
    with torch.no_grad():
        features = extractor(image_tensor)  # 输出形状[batch_size, 512, 7, 7]
    
    # 准备结果容器
    topk_features_list = []
    
    for i in range(batch_size):
        # 处理单张图像的特征
        img_features = features[i]  # [512, 7, 7]
        
        # 正确的重塑操作
        feature_vectors = img_features.permute(1, 2, 0).reshape(-1, 512)  # [49, 512]
        feature_vectors_cpu = feature_vectors.cpu().numpy()
        
        # 聚类
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(feature_vectors_cpu)
        
        # 获取TOP-K特征
        cluster_centers = kmeans.cluster_centers_  # [k, 512]
        cluster_counts = np.bincount(kmeans.labels_)
        top_indices = np.argsort(cluster_counts)[::-1][:k]
        top_k_centers = cluster_centers[top_indices]  # [k, 512]
        
        # 转换为张量并存储
        topk_features_tensor = torch.tensor(top_k_centers, device=device, dtype=torch.float32)
        topk_features_list.append(topk_features_tensor)
    
    # 堆叠batch结果
    return torch.stack(topk_features_list)  # [batch_size, k, 512]