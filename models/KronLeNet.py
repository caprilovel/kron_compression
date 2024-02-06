import numpy as np
import torch
import torch.nn as nn
from gkpd.tensorops import kron

class KronLinear(nn.Module):
    def __init__(self, a_shape, b_shape, structured_sparse=False, rank=None, bias=True) -> None:
        super().__init__()
        if not rank:
            rank = min(*a_shape, *b_shape) 
        self.rank = rank
        
        self.structured_sparse = structured_sparse
        if structured_sparse:
            self.s = nn.Parameter(torch.randn(rank, *a_shape), requires_grad=True)
        
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
        
        # a = self.s.unsqueeze(0) * self.a
        w = kron(self.a, self.b)
        
        out = x @ w 
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out


        
        
class KronLeNet(nn.Module):
    def __init__(self) -> None:
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
    
        self.kronfc1 = KronLinear(rank1, (16, 10), (16, 12), bias=False)
        
        self.kronfc2 = KronLinear(rank2, (10, 12), (12, 7), bias=False)
        self.kronfc3 = KronLinear(rank3, (12, 2), (7, 5), bias=False)
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
    
class BlockwiseKronLinear(nn.Module):
    def __init__(self, rank, a_shape, b_shape, bias=True) -> None:
        super().__init__()
        self.rank = rank
        self.s = nn.Parameter(torch.randn(*a_shape), requires_grad=True)
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
        
        a = self.s.unsqueeze(0) * self.a
        w = kron(a, self.b)
        
        out = x @ w 
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out
    


        
        
class BlockwiseKronLeNet(nn.Module):
    def __init__(self) -> None:
        super(BlockwiseKronLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        rank1 = 21
        rank2 = 10
        rank3 = 4
    
        self.kronfc1 = BlockwiseKronLinear(rank1, (16, 10), (16, 12), bias=False)
        
        self.kronfc2 = BlockwiseKronLinear(rank2, (10, 12), (12, 7), bias=False)
        self.kronfc3 = BlockwiseKronLinear(rank3, (12, 2), (7, 5), bias=False)
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
        



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


