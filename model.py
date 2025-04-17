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






def vec_avg(x, k=25, m=8, idx=None):
    """
    输入:
        x: (B, 3, N) 的点云张量
        k: 邻居数量
        m: 选择 top-m 个法向量用于平均
        idx: 可选的 knn 索引，形状为 (B, N, k)

    输出:
        normals: (B, N, 3)，每个点一个平均法向量
    """
    batch_size, num_dims, num_points = x.shape
    device = x.device

    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)，你需要实现 knn()

    normals = torch.zeros(batch_size, num_points, 3, device=device)

    for batch in range(batch_size):
        idx_batch = idx[batch]  # (N, k)
        for pt in range(num_points):
            center = x[batch, :, pt]  # (3,)
            tor_list = []

            for i in range(k):
                for j in range(i + 1, k):
                    ni = idx_batch[pt, i]
                    nj = idx_batch[pt, j]
                    x1 = x[batch, :, ni]
                    x2 = x[batch, :, nj]

                    vec1 = x1 - center
                    vec2 = x2 - center
                    normal = torch.cross(vec1, vec2, dim=0)
                    norm = torch.norm(normal)
                    if norm > 1e-6:
                        normal = normal / norm  # 单位化
                        tor_list.append(normal)

            # 如果三角形太少，填 0
            if len(tor_list) == 0:
                continue

            # 计算 pairwise 相似性矩阵
            n = len(tor_list)
            sim_scores = torch.zeros(n, device=device)
            for i in range(n):
                for j in range(n):
                    sim_scores[i] += torch.abs(torch.dot(tor_list[i], tor_list[j]))

            # 选出 top-m 最相似的法向量
            top_m_idx = torch.topk(sim_scores, min(m, len(tor_list)))[1]
            top_normals = torch.stack([tor_list[i] for i in top_m_idx], dim=0)  # (m, 3)

            avg_normal = torch.mean(top_normals, dim=0)
            avg_normal = avg_normal / (torch.norm(avg_normal) + 1e-6)  # 再单位化
            normals[batch, pt] = avg_normal

    return normals  # (B, N, 3)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x





class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(LightDGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.m = args.m

        # 第一阶段：EdgeConv 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 + 3, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # 第二阶段：EdgeConv 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 + 3, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # 第三阶段：EdgeConv 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 + 3, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        # 第四阶段：全局特征映射
        self.conv4 = nn.Sequential(
            nn.Conv1d(128 + 64 + 64, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.GELU()
        )

        # 分类头
        self.linear1 = nn.Linear(args.emb_dims, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size, _, num_points = x.size()

        # 平均法向量版本
        normals = vec_avg(x, k=self.k, m=self.m).permute(0, 2, 1)  # (B, 3, N)
        x1 = self.conv1(torch.cat([x, normals], dim=1).unsqueeze(-1)).squeeze(-1)

        normals2 = vec_avg(x1, k=self.k, m=self.m).permute(0, 2, 1)
        x2 = self.conv2(torch.cat([x1, normals2], dim=1).unsqueeze(-1)).squeeze(-1)

        normals3 = vec_avg(x2, k=self.k, m=self.m).permute(0, 2, 1)
        x3 = self.conv3(torch.cat([x2, normals3], dim=1).unsqueeze(-1)).squeeze(-1)

        x_all = torch.cat((x1, x2, x3), dim=1)
        x = self.conv4(x_all)

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = F.gelu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x
