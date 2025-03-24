
# 相似度测试

import torch
import torch.nn as nn

class SimAM(nn.Module):
    def __init__(self, lamda=1e-5):
        super().__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入张量的形状信息
        b, c, h, w = x.shape
        # 计算像素点数量
        n = h * w - 1
        # 计算输入张量在通道维度上的均值
        mean = torch.mean(x, dim=[-2, -1], keepdim=True)
        # 计算输入张量在通道维度上的方差
        var = torch.sum(torch.pow((x - mean), 2), dim=[-2, -1], keepdim=True) / n
        # 计算特征图的激活值
        e_t = torch.pow((x - mean), 2) / (4 * (var + self.lamda)) + 0.5
        # 使用 Sigmoid 函数进行归一化
        out = self.sigmoid(e_t) * x
        return out


# 测试模块
if __name__ == "__main__":
    # 创建 SimAM 实例并进行前向传播测试
    layer = SimAM(lamda=1e-5)
    x = torch.randn((2, 3, 224, 224))
    output = layer(x)
    print("Output shape:", output.shape)