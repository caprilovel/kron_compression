import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from models.KronLinear import KronLinear, LowRankLinear
import random

#using 5-fold of mnist
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)


train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

full_dataset = torch.utils.data.ConcatDataset([train_mnist, test_mnist])
kf = KFold(n_splits=5, shuffle=True)
batch_size = 64



class OneLayer(nn.Module):
    def __init__(self):
        super(OneLayer, self).__init__()
        self.linear = nn.Linear(784, 10)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = x.view(-1, 784)
        return self.activation(self.linear(x))

class KronLayer(nn.Module):
    def __init__(self, rank_rate=1):
        super(KronLayer, self).__init__()
        self.kronlinear = KronLinear(784, 10, structured_sparse=True, rank_rate=rank_rate)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = x.view(-1, 784)
        return self.activation(self.kronlinear(x))
    
class LowRankLayer(nn.Module):
    def __init__(self, rank_rate=1):
        super(LowRankLayer, self).__init__()
        self.lrlinear = LowRankLinear(784, 10, rank_rate=rank_rate)
        
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = x.view(-1, 784)
        return self.activation(self.lrlinear(x))

# calculate the parameter and flops 
model = OneLayer()
from deepspeed.profiling.flops_profiler import get_model_profile
flops, macs, params = get_model_profile(model, (1, 1, 28, 28))
print(flops, params)

import time 

# model = model.cuda()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# start = time.time()
# for epoch in range(50):
#     for i, (x, y) in enumerate(train_loader):
#         x = x.cuda()
#         y = y.cuda()
#         optimizer.zero_grad()
#         output = model(x)
#         loss = F.cross_entropy(output, y)
#         loss.backward()
#         optimizer.step()
#         if i % 100 == 0:
#             print(f'iteration {i}, loss {loss.item()}')
            
# end = time.time()
# print('training time', end - start)

# with torch.no_grad():
#     start = time.time()
#     correct = 0
#     total = 0
#     for i, (x, y) in enumerate(test_loader):
#         x = x.cuda()
#         y = y.cuda()
#         output = model(x)
#         _, predicted = torch.max(output, 1)
#         total += y.size(0)
#         correct += (predicted == y).sum().item()
#     print(f'accuracy: {correct/total}')
#     end = time.time()
#     print('testing time', end - start)

# s_num = torch.numel(model.kronlinear.s)
# sparse_num = torch.sum(model.kronlinear.s < 1e-5)
# print(f'sparse ratio: {sparse_num/s_num}') 

def train_test(model, ):
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    start = time.time()
    accuracy = []
    sparse_ratio = []
    for fold, (train_index, test_index) in enumerate(kf.split(full_dataset)):
        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_dataset, train_index), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_dataset, test_index), batch_size=batch_size, shuffle=True)
        prune_mask = torch.ones_like(model.linear.weight)
        for epoch in range(50):
            
            for i, (x, y) in enumerate(train_loader):
                x = x.cuda()
                y = y.cuda()
                optimizer.zero_grad()
                output = model(x)
                loss = F.cross_entropy(output, y)
                
                loss.backward()
                optimizer.step()
                # prune the weight
                new_prune_mask = torch.abs(model.linear.weight) > 1e-2
                prune_mask = torch.logical_and(prune_mask, new_prune_mask)
                model.linear.weight.data *= prune_mask
                if i % 100 == 0:
                    print(f'iteration {i}, loss {loss.item()}')
            print('sparse ratio:', torch.sum(prune_mask).item()/torch.numel(prune_mask))
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (x, y) in enumerate(test_loader):
                x = x.cuda()
                y = y.cuda()
                output = model(x)
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            print(f'fold {fold} accuracy: {correct/total}')
            accuracy.append(correct/total)
        sparse_ratio.append(torch.sum(prune_mask).item()/torch.numel(prune_mask))
    import numpy as np
    accuracy = np.array(accuracy)
    print(np.mean(accuracy), np.std(accuracy))
    print(np.mean(sparse_ratio), np.std(sparse_ratio))
    return accuracy
accuracy = []

model = OneLayer()
flops, macs, params = get_model_profile(model, (1, 1, 28, 28))
print(flops, params)
train_test(model)