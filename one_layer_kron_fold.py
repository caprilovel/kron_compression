import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from models.KronLinear import KronLinear, LowRankLinear
import random
from utils.decomposition import kron_decompose_model

#using 5-fold of mnist
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)


train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

full_dataset = torch.utils.data.ConcatDataset([train_mnist, test_mnist])
kf = KFold(n_splits=5, shuffle=True)
batch_size = 64



class MultiLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 10)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return self.activation(self.linear3(x))
        
    

class OneLayer(nn.Module):
    def __init__(self):
        super(OneLayer, self).__init__()
        self.linear = nn.Linear(784, 10)
        self.activation = nn.Sigmoid()
        
        
    def forward(self, x):
        x = x.view(-1, 784)
        return self.activation(self.linear(x))


    
class LowRankLayer(nn.Module):
    def __init__(self, rank_rate=1):
        super(LowRankLayer, self).__init__()
        self.lrlinear = LowRankLinear(784, 10, rank_rate=rank_rate)
        
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = x.view(-1, 784)
        return self.activation(self.lrlinear(x))

# calculate the parameter and flops 
model = MultiLayer()
layer_config = {
    'rank': 1,
    'structured_sparse': True,
    'shape_bias': -5,
}
model = kron_decompose_model(model, layer_config)
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
    sparse = []
    for fold, (train_index, test_index) in enumerate(kf.split(full_dataset)):
        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_dataset, train_index), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_dataset, test_index), batch_size=batch_size, shuffle=True)
        
        for epoch in range(50):
            for i, (x, y) in enumerate(train_loader):
                x = x.cuda()
                y = y.cuda()
                optimizer.zero_grad()
                output = model(x)
                loss = F.cross_entropy(output, y)
                # loss += torch.norm(model.kronlinear.s, p=1) * 0.0001
                loss.backward()
                optimizer.step()
                # if i % 100 == 0:
                #     print(f'iteration {i}, loss {loss.item()}')
        
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

    import numpy as np
    accuracy = np.array(accuracy)
    print(np.mean(accuracy), np.std(accuracy))
    
    print(np.mean(sparse), np.std(sparse))
    
    return accuracy
accuracy = []
variance = []
# for i in range(1, 40, 1):
#     rank_rate = i * 0.1
#     model = KronLayer(rank_rate=rank_rate)
#     flops, macs, params = get_model_profile(model, (1, 1, 28, 28))
#     print(flops, params)
#     result = train_test(model)
#     import numpy as np
#     accuracy.append(np.mean(result))
#     variance.append(np.std(result))
                    
# print(accuracy)
# print(variance)
# # save accuracy and variance 
# import pickle
# import os
# if not os.path.exists('accuracy.pkl'):
#     os.mknod('accuracy.pkl')
# with open('accuracy.pkl', 'wb') as f:
#     pickle.dump(accuracy, f)

# if not os.path.exists('variance.pkl'):
#     os.mknod('variance.pkl')
# with open('variance.pkl', 'wb') as f:
#     pickle.dump(variance, f)


model = MultiLayer()
model = kron_decompose_model(model, layer_config)
flops, macs, params = get_model_profile(model, (1, 1, 28, 28))
print(flops, params)
train_test(model)
