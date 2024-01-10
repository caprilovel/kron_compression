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

from models.KronLeNet import BlockwiseKronLeNet

def count_parameter(model):
    
    total = 0
    for i in model.parameters():
        total += i.numel()
    print(total)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BlockwiseKronLeNet().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)
count_parameter(model)
def calculate_sparsity(model, threshold=1e-6):
    total_params = 0
    sparse_params = 0

    for param in model.parameters():
        total_params += param.numel()  # 统计参数总数
        sparse_params += torch.sum(torch.abs(param) < threshold).item()  # 统计绝对值小于阈值的参数数量

    sparsity = sparse_params / total_params  # 计算稀疏性
    return sparsity, sparse_params, total_params

def train(model, train_loader, criterion, optimizer, epochs, l1_weight=0.01):
    for epoch in range(epochs):
        running_loss = 0.0
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


train(model, train_loader, criterion, optimizer, epochs=10)
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

test(model, test_loader)
torch.save(model.state_dict(), './model_save/blockwisekronlenet_mnist.pth')