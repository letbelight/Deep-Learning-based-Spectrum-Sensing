import torch
from torch import nn
from torch.nn import functional as  F
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# 从自定义模块中导入数据加载类，用于加载训练、测试和验证数据集
from SignalDataLoader import LoadSignal
# 从自定义模块中导入卷积神经网络模型
from ConNet import ConNet

def main():
    # 创建 LoadSignal 类的实例，加载训练集数据
    signal_train = LoadSignal('dataset', mode='train')
    # 创建 LoadSignal 类的实例，加载测试集数据
    signal_test = LoadSignal('dataset', mode='test')
    # 创建 LoadSignal 类的实例，加载验证集数据
    signal_val = LoadSignal('dataset', mode='val')

    # 使用 DataLoader 对训练集数据进行封装，设置批量大小为 32，打乱数据顺序，使用 1 个工作进程加载数据
    train_loader = DataLoader(signal_train, batch_size=32, shuffle=True,
                              num_workers=1)
    # 使用 DataLoader 对测试集数据进行封装，设置批量大小为 32，打乱数据顺序，使用 1 个工作进程加载数据
    test_loader = DataLoader(signal_test, batch_size=32, shuffle=True,
                              num_workers=1)
    # 使用 DataLoader 对验证集数据进行封装，设置批量大小为 32，打乱数据顺序，使用 1 个工作进程加载数据
    val_loader = DataLoader(signal_val, batch_size=32, shuffle=True,
                             num_workers=1)

    # 检查是否有可用的 CUDA 设备，如果有则使用 GPU 进行计算，否则使用 CPU
    device = torch.device('cuda')
    # 实例化卷积神经网络模型，并将其移动到指定设备（GPU）上
    net = ConNet().to(device)
    # 定义交叉熵损失函数，并将其移动到指定设备（GPU）上
    criteon = nn.CrossEntropyLoss().to(device)
    # 定义随机梯度下降（SGD）优化器，用于更新模型的参数
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # 定义学习率调度器，每 20 个 epoch 将学习率乘以 0.1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 初始化最佳准确率和对应的 epoch 数
    best_acc, best_epoc = 0, 0

    # 开始训练循环，共进行 50 个 epoch
    for epoch in range(50):
        # 将模型设置为训练模式
        net.train()
        # 遍历训练集中的每个批次
        for batch_idx, (x, label) in enumerate(train_loader):
            # 将输入数据和标签移动到指定设备（GPU）上
            x, label = x.to(device), label.to(device)
            # 将输入数据传入模型，得到模型的输出
            logits = net(x)
            # 计算损失值，使用交叉熵损失函数
            loss = criteon(logits, label)
            # 清空优化器中的梯度信息
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新模型的参数
            optimizer.step()

        # 打印当前 epoch 的损失值
        print('epoch:', epoch, 'loss:', loss.item())
        # 更新学习率
        scheduler.step()

        # 将模型设置为评估模式，关闭 Dropout 等训练时使用的特殊层
        net.eval()
        # 上下文管理器，用于关闭梯度计算，减少内存消耗
        with torch.no_grad():
            # 初始化正确预测的样本数
            total_correct = 0
            # 初始化样本总数
            total_num = 0
            # 遍历验证集中的每个批次
            for x, label in val_loader:
                # 将输入数据和标签移动到指定设备（GPU）上
                x, label = x.to(device), label.to(device)
                # 将输入数据传入模型，得到模型的输出
                logits = net(x)
                # 获取模型输出中概率最大的类别索引，作为预测结果
                pred = logits.argmax(dim=1)
                # 计算预测结果与真实标签相等的样本数，并累加到 total_correct 中
                total_correct += torch.eq(pred, label).float().sum().item()
                # 累加当前批次的样本数到 total_num 中
                total_num += x.size(0)

            # 计算验证集的准确率
            accuracy = total_correct / total_num

            # 如果当前准确率大于最佳准确率
            if accuracy > best_acc:
                # 更新最佳 epoch 数
                best_epoc = epoch
                # 更新最佳准确率
                best_acc = accuracy
                # 保存当前模型的参数到文件 'model.mdl' 中
                torch.save(net.state_dict(), 'model.mdl')

        # 打印最佳 epoch 数和最佳准确率
        print('best_epoch:', best_epoc, 'best_accuracy:', best_acc)

    # 初始化检测到的信号数量
    total_detection = 0
    # 初始化虚警的数量
    total_falsealarm = 0
    # 初始化有信号的样本数量
    y1 = 0
    # 初始化无信号的样本数量
    y0 = 0
    # 加载之前保存的最佳模型的参数
    net.load_state_dict(torch.load('model.mdl'))
    # 遍历测试集中的每个批次
    for x, label in test_loader:
        # 将输入数据和标签移动到指定设备（GPU）上
        x, label = x.to(device), label.to(device)
        # 获取当前批次中样本的数量
        l = len(label)
        # 遍历当前批次中的每个样本
        for i in range(l):
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
    # 调用 main 函数，开始执行训练和测试过程
    main()
