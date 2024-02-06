import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 

import numpy as np
import pandas as pds
from einops import rearrange, reduce, repeat

import torchvision 
from torchvision import transforms, datasets


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform) 

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


from models.KronViT import ViT

model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 100,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs, outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' % (epoch + 1, len(train_loader), running_loss / len(train_loader)))
    print('Finished Training')
    
train(model, train_loader, criterion, optimizer, 100)
def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))
    
test(model, test_loader)