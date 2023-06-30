import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")




import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
num_epochs = 25  # 50轮
batch_size = 50  # 50步长
learning_rate = 0.01  # 学习率0.01
from torch.utils.data import DataLoader
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    transform = transforms.Compose(
        [
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
        )
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=transform)
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=transform)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d
    
