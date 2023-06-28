# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
    

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, 64, 3)
        self.layer2 = self.make_layer(64, 128, 4, stride=2)
        self.layer3 = self.make_layer(128, 256, 6, stride=2)
        self.layer4 = self.make_layer(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)


def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='../cifar-10-python
/cifar-10-batches-py', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='../cifar-10-python
/cifar-10-batches-py', train=False, download=False, transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def main():
    # 创建神经网络模型实例
    model = NeuralNetwork()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 加载数据
    dataset_train, dataset_val, data_loader_train, data_loader_val = read_data()

    # 设置设备为 GPU 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader_train):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播、反向传播和优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计损失
            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    # 保存模型参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    torch.save(model.state_dict(), parent_dir + '/pth/model.pth')

    return model

    
