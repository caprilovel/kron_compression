from global_utils.torch_utils.cuda import find_gpus
find_gpus(1)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat
import torchvision
from torchvision import datasets, transforms

from gkpd.tensorops import kron
from time import time 

from global_utils.tools import random_seed
from global_utils.torch_utils.Args import TorchArgs

random_seed(2024)
parser = TorchArgs()
parser.add_argument('--group_id', type=int, default=0, help='group id')
parser.add_argument('--thresold', type=float, default=1e-1, help='thresold')

group_id = parser.parse_args().group_id
thresold = parser.parse_args().thresold

print(f"--------- hyper params -------")
print(f"group_id: {group_id}")
print(f"thresold: {thresold}")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

Kronnecker_group = [[
    [(16, 10 * 5), (16, 12 * 2)],
    [(10 * 5, 12 * 2), (12 * 2, 7 * 5)],
    [(12 * 2, 2), (7 * 5, 5)]
    ],
    [[(8, 10), (32, 12)],
     [(5, 6), (24, 14)],
     [(4, 2), (21, 5)]
     ],
    [[(32, 12), (8, 10)],
     [(24, 14), (5, 6)],
        [(21, 5), (4, 2)]
     ],
    [[(4, 5), (64, 24)],
     [(5, 3), (24, 28)],
     [(4, 2), (21, 5)]
     ],
    ]


class KronLinear(nn.Module):
    def __init__(self, rank, a_shape, b_shape, structured_sparse=False, bias=True) -> None:
        """Kronecker Linear Layer

        Args:
            rank (int): _description_
            a_shape (_type_): _description_
            b_shape (_type_): _description_
            structured_sparse (bool, optional): _description_. Defaults to False.
            bias (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.rank = rank
        self.structured_sparse = structured_sparse
        if structured_sparse:
            self.s = nn.Parameter(torch.randn( *a_shape), requires_grad=True)
        self.a = nn.Parameter(torch.randn(rank, *a_shape), requires_grad=True)
        self.b = nn.Parameter(torch.randn(rank, *b_shape), requires_grad=True)
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)
        bias_shape = np.multiply(a_shape, b_shape)
        if bias:
            self.bias = nn.Parameter(torch.randn(*bias_shape[1:]), requires_grad=True)
        else:
            self.bias = None
        
    def forward(self, x):
        a = self.a
        if self.structured_sparse:
            a = self.s.unsqueeze(0) * self.a
        
        # a = self.s.unsqueeze(0) * self.a
        w = kron(a, self.b)
        
        out = x @ w 
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out
    
class KronLeNet(nn.Module):
    def __init__(self, group_id=1) -> None:
        super(KronLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        rank1 = 21
        rank2 = 10
        rank3 = 4
        
        self.kronfc1 = KronLinear(rank1, Kronnecker_group[group_id][0][0], Kronnecker_group[group_id][0][1], bias=False, structured_sparse=True)
        
        self.kronfc2 = KronLinear(rank2, Kronnecker_group[group_id][1][0], Kronnecker_group[group_id][1][1], bias=False, structured_sparse=True)
        
        self.kronfc3 = KronLinear(rank3, Kronnecker_group[group_id][2][0], Kronnecker_group[group_id][2][1], bias=False, structured_sparse=True)
        self.relu3 = nn.LeakyReLU()
        self.relu4 = nn.LeakyReLU()


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu3(self.kronfc1(x))
        x = self.relu4(self.kronfc2(x))
        x = self.kronfc3(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models.LeNet import LeNet
model = LeNet().to(device)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def calculate_sparsity(model, threshold=1e-6):
    total_params = 0
    sparse_params = 0

    for param in model.parameters():
        total_params += param.numel()  # 统计参数总数
        sparse_params += torch.sum(torch.abs(param) < threshold).item()  # 统计绝对值小于阈值的参数数量

    sparsity = sparse_params / total_params  # 计算稀疏性
    return sparsity, sparse_params, total_params

from global_utils.torch_utils.log_utils import train_log
@train_log()
def train(model, train_loader, criterion, optimizer, epochs, l1_weight=0.01, thresold=1e-1):
    
    i_time = time()
    decay_weight = [1, 0.1, 0.01, 0.001, 0.0001]
    weight1 = l1_weight

    thresold = thresold

    for epoch in range(epochs):
        
        running_loss = 0.0
        l1_weight = decay_weight[epoch//20] * weight1
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs, outputs.shape)
            loss = criterion(outputs, labels)

            
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    print("Finished Training, total_time:", time()-i_time)
init_time = time()
        
train(model, train_loader, criterion, optimizer, epochs=100, thresold=thresold)
training_time = time()

def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    print(f"Accuracy: {100 * correct/total}")
    return 100 * correct/total
    
accuracy = test(model, test_loader)
print("inference time:", time()-training_time)

with open('/home/zhu.3723/kron_compression/result.txt') as f:
    f.write(f"--------- hyper params -------\n")
    f.write(f"group_id: {group_id}\n")
    f.write(f'group:{Kronnecker_group[group_id]}')
    f.write(f"thresold: {thresold}\n")
    f.write(f"--------- training result -------\n")
    f.write(f"training time: {training_time-init_time}\n")
    f.write(f"--------- inference result -------\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"inference time: {time()-training_time}\n")
    f.write(f"----------------------------------\n")    

