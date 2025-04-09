import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


 
def seq_dict(dictionary, m):
    # 获取字典的 keys 和 values
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    # 将 values 转换为 tensor
    values_tensor = torch.tensor(values, dtype=torch.float32)  # 确保是浮点类型

    # 使用 topk 获取最大的 m 个值的索引
    _, indexes = torch.topk(values_tensor, m)

    # 返回索引对应的 keys
    return [keys[i] for i in indexes]






def vec(x, k=25, m=8, idx=None):
    batch_size, num_dims, num_points = x.shape

    if idx is None:
        idx = knn(x, k=k)  # 需要你提前实现 knn 函数

    dic = {}

    for batch in range(batch_size):
        idx_batch = idx[batch]  # (num_points, k)

        for pt in range(num_points):
            tor_dict = {}
            center = x[batch, :, pt]  # (num_dims,)

            # 遍历所有邻居对，计算三角形法向量
            for i in range(k):
                for j in range(i + 1, k):  # i < j 避免重复
                    ni = idx_batch[pt, i]  # 第 i 个邻居索引
                    nj = idx_batch[pt, j]  # 第 j 个邻居索引
                    x1 = x[batch, :, ni]
                    x2 = x[batch, :, nj]

                    vec1 = x1 - center
                    vec2 = x2 - center
                    normal = torch.cross(vec1, vec2, dim=0)

                    tor_dict[(int(ni), int(nj))] = normal

            # 计算每个法向量与其他法向量的夹角相似性（点积）
            another_dict = {}
            for key1 in tor_dict:
                s = 0
                for key2 in tor_dict:
                    n1 = tor_dict[key1]
                    n2 = tor_dict[key2]
                    s += torch.abs(torch.dot(n1, n2))
                another_dict[key1] = s

            # 保留 top-m 相似度的三角形
            sorted_keys = sorted(another_dict.items(), key=lambda x: x[1], reverse=True)
            top_keys = set([item[0] for item in sorted_keys[:m]])
            tor_dict = {k: v for k, v in tor_dict.items() if k in top_keys}

            dic[(batch, pt)] = tor_dict

    return dic


def vec_to_tensor(dic, batch_size, num_points, m, num_dims=3):
    features = torch.zeros(batch_size, num_points, m * num_dims)

    for batch in range(batch_size):
        for pt in range(num_points):
            tor_dict = dic.get((batch, pt), {})
            
            # 初始化法向量池
            normal_pool = torch.zeros(m, num_dims)
            
            # 按相似度选择 top-m 法向量
            sorted_keys = sorted(tor_dict.items(), key=lambda x: torch.abs(torch.dot(x[1], x[1])), reverse=True)
            top_keys = [key[0] for key in sorted_keys[:m]]
            
            # 填充法向量池
            for i, (key, normal) in enumerate(tor_dict.items()):
                if key in top_keys:
                    index = top_keys.index(key)
                    normal_pool[index] = normal

            # 将法向量池展平并添加到特征矩阵
            features[batch, pt] = normal_pool.view(-1)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        
        # 卷积层：减少了卷积层数目，只保留了 3 个卷积层
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)  # 输入 3 个通道（x, y, z），输出 64 个通道
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)  # 输入 64 个通道，输出 64 个通道
        self.conv3 = nn.Conv1d(64, args.emb_dims, kernel_size=1, bias=False)  # 输出嵌入维度的特征
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(args.emb_dims)
        
        # 全连接层：减小神经元数量
        self.linear1 = nn.Linear(args.emb_dims, 256, bias=False)  # 从嵌入维度到 256
        self.bn4 = nn.BatchNorm1d(256)  # 对全连接层进行批归一化
        self.dp1 = nn.Dropout(0.3)  # 使用 Dropout，减少过拟合
        self.linear2 = nn.Linear(256, output_channels)  # 最终输出层，用于分类（或者回归）

    def forward(self, x):
        # 卷积层 + 批归一化 + 激活函数（ReLU）
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 全局池化：池化操作对每个样本的点云进行聚合
        x = F.adaptive_max_pool1d(x, 1).squeeze()  # 使用最大池化得到全局特征

        # 全连接层 + 激活函数
        x = F.relu(self.bn4(self.linear1(x)))  # 全连接层 + 激活函数
        x = self.dp1(x)  # Dropout
        x = self.linear2(x)  # 输出层
        
        return x

class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(LightDGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.m = args.m

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3 + 3 * self.m, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128 + 64, args.emb_dims, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(args.emb_dims, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

    # 第一层
        dic1 = vec(x, k=self.k, m=self.m)
        feat1 = vec_to_tensor(dic1, batch_size, num_points, self.m).permute(0, 2, 1)
        x = torch.cat([x, feat1], dim=1)
        x1 = self.conv1(x)  # [B, 64, N]
    
    # 第二层
        dic2 = vec(x1, k=self.k, m=self.m)
        feat2 = vec_to_tensor(dic2, batch_size, num_points, self.m).permute(0, 2, 1)
        x2 = self.conv2(torch.cat([x1, feat2], dim=1).unsqueeze(-1)).squeeze(-1)  # [B, 128, N]

    # 第三层
        dic3 = vec(x2, k=self.k, m=self.m)
        feat3 = vec_to_tensor(dic3, batch_size, num_points, self.m).permute(0, 2, 1)
        x3 = self.conv3(torch.cat([x2, feat3], dim=1))  # [B, emb_dims, N]

    # 池化 & 分类头
        x = F.adaptive_max_pool1d(x3, 1).squeeze(-1)
        x = F.leaky_relu(self.bn4(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = self.linear2(x)
        return x




