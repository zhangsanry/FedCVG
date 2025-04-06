import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    用于CIFAR10的CNN网络
    输入: 3x32x32 (CIFAR10数据集)
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # 3通道输入，32个输出通道，3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, 3)  # 32通道输入，64个输出通道，3x3卷积核
        self.conv3 = nn.Conv2d(64, 64, 3)  # 64通道输入，64个输出通道，3x3卷积核
        
        # 池化后的特征图大小计算：
        # 原始图像: 32x32
        # conv1: 30x30 -> maxpool: 15x15
        # conv2: 13x13 -> maxpool: 6x6
        # conv3: 4x4 -> maxpool: 2x2
        self.fc1 = nn.Linear(64 * 2 * 2, 64)  # 全连接层
        self.fc2 = nn.Linear(64, 10)  # 输出层，10个类别
        
        self.dropout = nn.Dropout(0.25)  # dropout层，防止过拟合
        
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        # 第二个卷积块
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        # 第三个卷积块
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        # 展平操作
        x = x.view(-1, 64 * 2 * 2)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 