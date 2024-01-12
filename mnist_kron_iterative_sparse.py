import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat
import torchvision
from torchvision import datasets, transforms
import torch 
from gkpd import gkpd, KroneckerConv2d
from gkpd.tensorops import kron

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

from models.KronLinear import KronLinear 
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
        
        self.kronfc1 = KronLinear(rank1, (16, 10), (16, 12), bias=False, structured_sparse=True)
        
        self.kronfc2 = KronLinear(rank2, (10, 12), (12, 7), bias=False, structured_sparse=True)
        self.kronfc3 = KronLinear(rank3, (12, 2), (7, 5), bias=False, structured_sparse=True)
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
model = KronLeNet().to(device)



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


def train(model, train_loader, criterion, optimizer, epochs, l1_weight=0.01):
    decay_weight = [1, 0.1, 0.01, 0.001, 0.0001]
    weight1 = l1_weight
    mask1 = torch.ones_like(model.kronfc1.s)
    mask2 = torch.ones_like(model.kronfc2.s)
    mask3 = torch.ones_like(model.kronfc3.s)

    mask1, mask2, mask3 = mask1.to(device), mask2.to(device), mask3.to(device)    
    for epoch in range(epochs):
        if epoch % 5 == 0:
            mask1 = mask1 * (torch.abs(model.kronfc1.s) > 1e-4).float()
            model.kronfc1.s.data = model.kronfc1.s.data * mask1
            mask2 = mask2 * (torch.abs(model.kronfc2.s) > 1e-4).float()
            model.kronfc2.s.data = model.kronfc2.s.data * mask2
            mask3 = mask3 * (torch.abs(model.kronfc3.s) > 1e-4).float()
            model.kronfc3.s.data = model.kronfc3.s.data * mask3
            
        running_loss = 0.0
        l1_weight = decay_weight[epoch//20] * weight1
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs, outputs.shape)
            loss = criterion(outputs, labels)
            loss += l1_weight * torch.norm(model.kronfc1.s, p=1)
            loss += l1_weight * torch.norm(model.kronfc2.s, p=1)
            loss += l1_weight * torch.norm(model.kronfc3.s, p=1)
            
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss/len(train_loader)}")
        # print the sparsity of a, dont use == use the abs less than 1e-5
        print(calculate_sparsity(model))
        # calcu the s sparsity
        fc1_sparse = torch.sum(torch.abs(model.kronfc1.s) < 1e-5).item() 
        fc2_sparse = torch.sum(torch.abs(model.kronfc2.s) < 1e-5).item() 
        fc3_sparse = torch.sum(torch.abs(model.kronfc3.s) < 1e-5).item() 
        # total_params = model.kronfc1.s.numel() + model.kronfc2.s.numel() + model.kronfc3.s.numel()
        
        print(f"fc1 sparsity: {fc1_sparse}, fc2 sparsity: {fc2_sparse}, fc3 sparsity: {fc3_sparse}")
        print(f"total sparse params: {fc1_sparse + fc2_sparse + fc3_sparse}")
        # print(f"fc1 total params: {model.kronfc1.s.numel()}, fc2 total params: {model.kronfc2.s.numel()}, fc3 total params: {model.kronfc3.s.numel()}")
        # print(f"total params: {total_params}")
        
train(model, train_loader, criterion, optimizer, epochs=100)


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
    
print(test(model, test_loader))
# if __name__ == '__main__':
#     model = KronLinear(rank=10, a_shape=(16, 10), b_shape=(16, 12), bias=False)
#     a = torch.randn(184, 24, 256)
#     print(model(a).shape)
    
    

