# 这个是双注意力机制用于图像的重构

# CBAM是一个通用的轻量级的模块，可以用于多种网络
# CBAM模块的核心思想是对输入特征图进行两阶段的精炼：
# 首先通过通道注意力模块关注于“哪些通道是重要的”，然后通过空间注意力模块关注于“在哪里”是一个有信息的部分。
# 这种双重注意力机制使CBAM能够全面捕获特征中的关键信息。

# 通道注意力模块：
# Mc(F)=σ(MLP(AvgPool(F))+MLP(MaxPool(F)))
# F:输入特征图  AvgPool全局平均池化和MaxPool最大池化操作  MLP:多层感知机  σ：激活函数

# 空间注意力模块：
# Ms(F)=σ(f7×7([AvgPool(F);MaxPool(F)]))
# f7×7:是一个7×7的卷积操作 [AvgPool(F);MaxPool(F)]：将全局平均池化和最大池化操作拼接起来

# CBAM通过结合通道注意力和空间注意力，实现了对输入特征的双重精炼。这种设计使模型能够同时关注哪些通道和哪些空间位置是有意义的，从而提高了模型的表征能力和决策准确性
# CBAM能够根据任务需求和内容上下文动态地调整特征图中每个通道和空间位置的重要性
# 尽管CBAM为模型引入了额外的计算量，但其设计考虑了计算效率。通过全局池化和简单的卷积操作，CBAM能够在保持较低额外计算成本的同时带来性能提升。

import numpy as np
import torch
from torch import nn
from torch.nn import init

# 管道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



# CBAMBlock块
class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    kernel_size=input.shape[2]
    cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
    output=cbam(input)
    print(output.shape)

    