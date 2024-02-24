from utils.parser import Args
parser = Args().get_parser()
rank_rate = parser.rank_rate
structure_sparse = parser.ss
shape_bias = parser.shape_bias
model_type = parser.model
dataset = parser.dataset
device_setting = parser.device

kron_flag = parser.kron

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from time import time

from global_utils.torch_utils.log_utils import Logger

from utils.decomposition import kron_decompose_model 
from utils.loss import s_l1

logger = Logger()
logger.start_capture()
from torchsummary import summary
from models.KronLinear import KronLeNet_5
from data_factory.data_factory import loader_generate


# Dataset Setting

print(dataset)
train_loader, test_loader = loader_generate(dataset)
if dataset == 'mnist':
    input_size = (1, 1, 28, 28)
elif dataset.lower() == 'cifar10':
    input_size = (1, 3, 224, 224)

device = torch.device('cuda' if torch.cuda.is_available() and device_setting=='cuda' else 'cpu')
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator


# Model Generation
kron_config = {
    'rank_rate': rank_rate,
    'structured_sparse': structure_sparse,
    'bias':False,
    'shape_bias':shape_bias
}


if model_type == 'KronLeNet_5':
    model = KronLeNet_5(kron_config=kron_config).to(device)
elif model_type == 'LeNet_5':
    from models.LeNet import LeNet_5
    model = LeNet_5().to(device)
elif model_type.lower() == 'vit':
    from timm.models.vision_transformer import VisionTransformer
    import timm
    model = timm.create_model('vit_small_patch16_224', pretrained=False)
elif model_type.lower() == 'kronvit':
    import timm
    model = timm.create_model('vit_small_patch16_224', pretrained=False)
    from utils.decomposition import kron_decompose_model
    model = kron_decompose_model(model, kron_config)

if kron_flag: 
    model = kron_decompose_model(model, kron_config)
    print("kron decomposition")

print(input_size)
print(dataset)
#show the model FLOPs and params
# summary(model, input_size[1:], device=device_setting)

with get_accelerator().device():
    profile = get_model_profile(model, input_shape=input_size, )

from utils.evaluation import calcu_params, calculate_sparsity
calcu_params(model)
model = model.to(device)
# Training period

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


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
            loss = criterion(outputs, labels)
            if kron_flag:
                loss += s_l1(model, l1_weight)
            
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss/len(train_loader)}")
        
    torch.save(model.state_dict(), f'./model_save/mnist_kron_sparse_{model_type}.pth')
    print("Finished Training, total_time:", time()-i_time)
    if kron_flag:
        from utils.evaluation import calculate_sparsity
        print(calculate_sparsity(model))
    if model_type == 'KronLeNet_5':
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








