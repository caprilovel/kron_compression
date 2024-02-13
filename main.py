from utils.parser import Args
parser = Args().get_parser()
rank_rate = parser.rank_rate
structure_sparse = parser.ss
shape_bias = parser.shape_bias
 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from time import time

from global_utils.torch_utils.log_utils import Logger

logger = Logger()
logger.start_capture()
from torchsummary import summary
from models.KronLinear import KronLeNet_5
from data_factory.data_factory import loader_generate


dataset = parser.dataset


train_loader, test_loader = loader_generate(dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





kron_config = {
    'rank_rate': rank_rate,
    'structured_sparse': structure_sparse,
    'bias':False,
    'shape_bias':shape_bias
}


model = KronLeNet_5(kron_config=kron_config).to(device)
summary(model, (1, 28, 28))


from utils.evaluation import calcu_params, calculate_sparsity
calcu_params(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, train_loader, criterion, optimizer, epochs, l1_weight=0.01, thresold=1e-1):
    
    i_time = time()
    decay_weight = [1, 0.1, 0.01, 0.001, 0.0001]
    weight1 = l1_weight
    # mask1 = torch.ones_like(model.kronfc1.s)
    # mask2 = torch.ones_like(model.kronfc2.s)
    # mask3 = torch.ones_like(model.kronfc3.s)
    thresold = thresold
    # mask1, mask2, mask3 = mask1.to(device), mask2.to(device), mask3.to(device)    
    for epoch in range(epochs):
        # if epoch % 5 == 0:
            # mask1 = mask1 * (torch.abs(model.kronfc1.s) > thresold).float()
            # model.kronfc1.s.data = model.kronfc1.s.data * mask1
            # mask2 = mask2 * (torch.abs(model.kronfc2.s) > thresold).float()
            # model.kronfc2.s.data = model.kronfc2.s.data * mask2
            # mask3 = mask3 * (torch.abs(model.kronfc3.s) > thresold).float()
            # model.kronfc3.s.data = model.kronfc3.s.data * mask3
            # # if mask have 0 in any position, print mask 
            # print(model.kronfc1.s.data)
            # print(model.kronfc2.s.data)
            # print(model.kronfc3.s.data)
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

        
        # print(f"fc1py sparsity: {fc1_sparse}, fc2 sparsity: {fc2_sparse}, fc3 sparsity: {fc3_sparse}")
        # print(f"total sparse params: {fc1_sparse + fc2_sparse + fc3_sparse}")
        # print(f"fc1 total params: {model.kronfc1.s.numel()}, fc2 total params: {model.kronfc2.s.numel()}, fc3 total params: {model.kronfc3.s.numel()}")
        # print(f"total params: {total_params}")
    torch.save(model.state_dict(), f'./model_save/mnist_kron_sparse_.pth')
    print("Finished Training, total_time:", time()-i_time)
    print(calculate_sparsity(model))
    fc1_sparse = torch.sum(torch.abs(model.kronfc1.s) < 1e-5).item() 
    fc2_sparse = torch.sum(torch.abs(model.kronfc2.s) < 1e-5).item() 
    fc3_sparse = torch.sum(torch.abs(model.kronfc3.s) < 1e-5).item() 
    print('total sparsity rate:', (fc1_sparse + fc2_sparse + fc3_sparse)/(model.kronfc1.s.numel() + model.kronfc2.s.numel() + model.kronfc3.s.numel()))
init_time = time()
        
train(model, train_loader, criterion, optimizer, epochs=100, thresold=0.1)
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
logger.stop_capture()








