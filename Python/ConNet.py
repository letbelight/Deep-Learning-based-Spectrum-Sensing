import torch
# 导入 PyTorch 的神经网络模块
from torch import nn


# 定义一个名为 ConNet 的卷积神经网络类，继承自 nn.Module
class ConNet(nn.Module):

    def __init__(self):
        # 调用父类 nn.Module 的构造函数
        super(ConNet, self).__init__()

        # 定义卷积层单元，使用 nn.Sequential 按顺序组合多个层
        self.conv_unit = nn.Sequential(
            # 输入通道数为 1，输出通道数为 4，卷积核大小为 2，步长为 1，填充为 1
            # 输入形状: [b, 1, 64, 4]
            nn.Conv2d(1, 4, kernel_size=2, stride=1, padding=1),

            # 最大池化层，池化核大小为 2，步长为 1，填充为 0
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            # 输入通道数为 4，输出通道数为 8，卷积核大小为 2，步长为 2，填充为 1
            nn.Conv2d(4, 8, kernel_size=2, stride=2, padding=1),
            # 
            # 最大池化层，池化核大小为 2，步长为 1，填充为 0
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            # 输出形状: [b, 8, 32, 2]
        )

        # 以下代码块用于测试卷积层的输出形状，已注释掉
        # tmp = torch.randn(2, 1, 64, 4)
        # out = self.conv_unit(tmp)
        # print('layer 1', out.shape)

        # 定义全连接层单元，使用 nn.Sequential 按顺序组合多个层
        self.fc_unit = nn.Sequential(
            # 输入特征数为 8 * 32 * 2，输出特征数为 128
            nn.Linear(8 * 32 * 2, 128),
            # 使用 Sigmoid 激活函数
            nn.Sigmoid(),
            # 输入特征数为 128，输出特征数为 84
            nn.Linear(128, 84),
            # 使用 Sigmoid 激活函数
            nn.Sigmoid(),
            # 输入特征数为 84，输出特征数为 48
            nn.Linear(84, 48),
            # 使用 Sigmoid 激活函数
            nn.Sigmoid(),
            # 输入特征数为 48，输出特征数为 2
            nn.Linear(48, 2)
        )

    def forward(self, x):
        # 获取输入数据的批次大小
        batchsz = x.size(0)
        # 输入形状: [b, 1, 64, 4]
        # 通过卷积层单元进行前向传播
        x = self.conv_unit(x)
        # 输出形状: [b, 8, 32, 2]
        # 将卷积层的输出展平为一维向量
        x = x.view(batchsz, 8 * 32 * 2)
        # 形状: [b, 8 * 32 * 2]
        # 通过全连接层单元进行前向传播
        x = self.fc_unit(x)

        return x

def main():
    # 创建 ConNet 模型的实例
    net = ConNet()
    # 生成一个随机输入张量，形状为 [2, 1, 64, 4]
    tmp = torch.randn(2, 1, 64, 4)
    # 将随机输入张量传入模型进行前向传播
    out = net(tmp)
    # 打印卷积层的输出形状
    print('conv out', out.shape)
    # 打印模型的结构
    print(net)


if __name__ == '__main__':
    # 调用 main 函数
    main()
