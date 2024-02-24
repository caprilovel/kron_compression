from utils.decomposition import kron_decompose_model



import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import warnings
warnings.filterwarnings("ignore")
import timm
from timm.models.vision_transformer import VisionTransformer

from models.KronLinear import KronLinear
from utils.decomposition import linear2kronlinear


from models.LeNet import LeNet
    

kronLinear = kron_decompose_model(LeNet())
model = kronLinear

# train vit 1 epoch in cifar10, calculating the training time 
optimzier = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
from torchvision import datasets, transforms
transform = transforms.Compose(
    [transforms.ToTensor(),])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
import time
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
with get_accelerator().device():
    profile = get_model_profile(model, input_shape=(1, 1, 28, 28), )


from utils.evaluation import calcu_params, calculate_sparsity
# calcu_params(vit)

# start = time.time()
# model = vit 
# model = model.cuda()
# for i in range(1):
#     for data, target in train_loader:
#         data, target = data.cuda(), target.cuda()
#         optimzier.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimzier.step()
        
# end = time.time()
# print(f"training time: {end - start}")

# # inference time for 1 epoch in cifar10
# start = time.time()
# with torch.no_grad():
#     for i in range(1):
#         for data, target in train_loader:
#             data, target = data.cuda(), target.cuda()
#             output = model(data)
# end = time.time()
# print(f"inference time: {end - start}")

# train kron_resnet 1 epoch in cifar10, calculating the training time

model = model.cuda()


with get_accelerator().device():
    profile = get_model_profile(model, input_shape=(1, 1, 28, 28), )
calcu_params(model)


start = time.time()
for i in range(1):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimzier.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        from utils.loss import s_l1
        loss += s_l1(model, 1)
        print(s_l1(model))
        loss.backward()
        optimzier.step()

end = time.time()
print(f"training time: {end - start}")

# inference time for 1 epoch in cifar10
start = time.time()
with torch.no_grad():
    for i in range(1):
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            
end = time.time()
print(f"inference time: {end - start}")
from utils.evaluation import calculate_sparsity
print(calculate_sparsity(model))
print(model.fc1.s.shape, model.fc2.s.shape, model.fc3.s.shape)
