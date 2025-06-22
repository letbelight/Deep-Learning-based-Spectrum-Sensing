import torch
from torch.utils.data import DataLoader

# 从自定义的 SignalLoader_Test 模块中导入 LoadSignal 类，用于加载测试数据集
from SignalLoader_Test import LoadSignal
# 从自定义的 ConNet 模块中导入 ConNet 类，即卷积神经网络模型
from ConNet import ConNet


def main():
    # 创建一个 LoadSignal 类的实例，用于加载指定路径下的测试信号数据
    SignalData = LoadSignal('Signal_Test/SNR-16')
    # 使用 DataLoader 对测试数据集进行封装，设置批量大小为 32，打乱数据顺序，使用 1 个工作进程加载数据
    data_loader = DataLoader(SignalData, batch_size=32, shuffle=True,
                             num_workers=1)

    # 检查是否有可用的 CUDA 设备，如果有则使用 GPU 进行计算，否则使用 CPU
    device = torch.device('cuda')

    # 实例化卷积神经网络模型，并将其移动到指定设备（GPU）上
    net = ConNet().to(device)

    # 初始化检测到的信号数量
    total_detection = 0
    # 初始化虚警的数量
    total_falsealarm = 0
    # 初始化有信号的样本数量
    y1 = 0
    # 初始化无信号的样本数量
    y0 = 0
    # 加载之前训练好的模型参数
    net.load_state_dict(torch.load('model.mdl'))

    # 遍历测试数据集中的每个批次
    for x, label in data_loader:
        # 将输入数据和标签移动到指定设备（GPU）上
        x, label = x.to(device), label.to(device)
        # 获取当前批次中样本的数量
        l = len(label)
        # 遍历当前批次中的每个样本
        for i in range(l):
            # 打印当前样本的标签（这里被注释掉了）
            # print('label[i]',label[i])
            # 如果当前样本的标签为 1，表示有信号
            if label[i] == 1:
                # 获取当前样本的输入数据
                x_input = x[i]
                # 在输入数据的第 0 维增加一个维度，使其符合模型输入的批量维度要求
                x_input = torch.unsqueeze(x_input, dim=0)
                # 将输入数据传入模型，得到模型的输出
                out = net(x_input)
                # 获取模型输出中概率最大的类别索引，作为预测结果
                pred_1 = out.argmax(dim=1)
                # 计算预测结果与真实标签是否相等，相等则表示检测到信号，将结果转换为浮点数并求和
                detection = pred_1.eq(label[i]).sum().float().item()
                # 打印检测结果和预测结果（这里被注释掉了）
                # print('detection',detection,'pred',pred_1)
                # 累加检测到的信号数量
                total_detection += detection
                # 累加有信号的样本数量
                y1 += 1
            # 如果当前样本的标签为 0，表示无信号
            if label[i] == 0:
                # 获取当前样本的输入数据
                x_input = x[i]
                # 在输入数据的第 0 维增加一个维度，使其符合模型输入的批量维度要求
                x_input = torch.unsqueeze(x_input, dim=0)
                # 将输入数据传入模型，得到模型的输出
                out = net(x_input)
                # 获取模型输出中概率最大的类别索引，作为预测结果
                pred_0 = out.argmax(dim=1)
                # 计算预测结果与真实标签是否不相等，不相等则表示虚警，将结果转换为浮点数并求和
                falsealarm = pred_0.ne(label[i]).sum().float().item()
                # 打印虚警结果和预测结果（这里被注释掉了）
                # print(falsealarm,pred_0)
                # 累加虚警的数量
                total_falsealarm += falsealarm
                # 累加无信号的样本数量
                y0 += 1
    # 打印有信号的样本数量和无信号的样本数量
    print('number of H1:', y1, 'number of H0:', y0)
    # 打印检测到的信号数量和虚警的数量
    print('number of detection:', total_detection, 'number of false alarm:', total_falsealarm)
    # 计算检测概率，即检测到的信号数量除以有信号的样本数量
    Pd = total_detection / y1
    # 计算虚警概率，即虚警的数量除以无信号的样本数量
    Pf = total_falsealarm / y0
    # 打印检测概率
    print('Pd:', Pd)
    # 打印虚警概率
    print('Pf:', Pf)


if __name__ == '__main__':
    # 调用 main 函数，开始执行测试过程
    main()
