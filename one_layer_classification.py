import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from models.KronLinear import KronLinear

#using 5-fold of mnist

train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_mnist, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_mnist, batch_size=64, shuffle=False)

class OneLayer(nn.Module):
    def __init__(self):
        super(OneLayer, self).__init__()
        self.linear = nn.Linear(784, 10)
        self.activation = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.view(-1, 784)
        return self.activation(self.linear(x))

class KronLayer(nn.Module):
    def __init__(self, rank_rate=1):
        super(KronLayer, self).__init__()
        self.kronlinear = KronLinear(784, 10, structured_sparse=True, rank_rate=rank_rate)
        self.activation = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.view(-1, 784)
        return self.activation(self.kronlinear(x))



# calculate the parameter and flops 
model = KronLayer()
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
    for epoch in range(50):
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'iteration {i}, loss {loss.item()}')
                
    end = time.time()
    print('training time', end - start)

    with torch.no_grad():
        start = time.time()
        correct = 0
        total = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print(f'accuracy: {correct/total}')
        end = time.time()
        print('testing time', end - start)

    s_num = torch.numel(model.kronlinear.s)
    sparse_num = torch.sum(model.kronlinear.s < 1e-5)
    print(f'sparse ratio: {sparse_num/s_num}') 
    return correct/total
accuracy = []
for i in range(1, 20, 1):
    rank_rate = i * 0.1
    model = KronLayer(rank_rate=rank_rate)
    flops, macs, params = get_model_profile(model, (1, 1, 28, 28))
    print(flops, params)
    accuracy.append(train_test(model))
print(accuracy)