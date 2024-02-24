from global_utils.torch_utils.cuda import find_gpus
find_gpus(1)
from utils.decomposition import kron_decompose_model



import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import warnings
warnings.filterwarnings("ignore")
import timm
from timm.models.vision_transformer import VisionTransformer

kron_config = {'rank_rate': 0.1, 'structured_sparse': True, 'bias': False, 'shape_bias': 0}
from models.KronLinear import KronLinear
from utils.decomposition import linear2kronlinear


model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
model = kron_decompose_model(model, kron_config)
# 1.The original network

# train vit 1 epoch in cifar10, calculating the training time 
optimzier = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
from torchvision import datasets, transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Resize((224,224))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
import time
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
with get_accelerator().device():
    profile = get_model_profile(model, input_shape=(1, 3, 224, 224), )

from utils.evaluation import calcu_params, calculate_sparsity
calcu_params(model)

def freeze_linear(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_linear(module)
        else:
            if isinstance(module, nn.Linear):
                module.requires_grad = False
            else:
                continue

# freeze_linear(vit)
model = model.cuda()
start = time.time()
for i in range(1):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimzier.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimzier.step()
        
end = time.time()
print("original model")
print(f" training time: {end - start}")




# inference time for 1 epoch in cifar10
start = time.time()
with torch.no_grad():
    for i in range(1):
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
end = time.time()
print(f"inference time: {end - start}")



# 2. Kronecker decomposed network freeze a

# train kron_resnet 1 epoch in cifar10, calculating the training time
model = kron_decompose_model(model, kron_config)
# kron_resnet = vit
model = model.cuda()
# with get_accelerator().device():
#     profile = get_model_profile(kron_resnet, input_shape=(1, 3, 224, 224), )
calcu_params(model)

def freeze_a(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_a(module)
        else:
            if isinstance(module, KronLinear):
                module.a.requires_grad = False
            else:
                continue

def freeze_b(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_b(module)
        else:
            if isinstance(module, KronLinear):
                module.b.requires_grad = False
            else:
                continue
            
freeze_a(model)
# freeze_b(kron_resnet)

start = time.time()
for i in range(1):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimzier.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        from utils.loss import s_l1
        loss += s_l1(model, 0.1)
        loss.backward()
        optimzier.step()

end = time.time()
print(f"freezed A kronecker model")
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


# 3. Kronecker decomposed network freeze B

model = timm.create_model('vit_base_patch32_224', pretrained=True)
model = kron_decompose_model(model, kron_config)
model = model.cuda()
calcu_params(model)

            
freeze_b(model)

start = time.time()
for i in range(1):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimzier.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        from utils.loss import s_l1
        loss += s_l1(model, 0.1)
        loss.backward()
        optimzier.step()

end = time.time()
print(f"freezed B kronecker model")
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


# 4. Kronecker decomposed network freeze AB

# train kron_resnet 1 epoch in cifar10, calculating the training time
model = timm.create_model('vit_base_patch32_224', pretrained=True)
model = kron_decompose_model(model, kron_config)
model = model.cuda()
calcu_params(model)

freeze_a(model)
freeze_b(model)

start = time.time()
for i in range(1):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimzier.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        from utils.loss import s_l1
        loss += s_l1(model, 0.1)
        loss.backward()
        optimzier.step()

end = time.time()
print(f"freezed A and B kronecker model")
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

# 5. Kronecker decomposed network freeze None

# train kron_resnet 1 epoch in cifar10, calculating the training time
model = timm.create_model('vit_base_patch32_224', pretrained=True)
model = kron_decompose_model(model, kron_config)
# kron_resnet = vit
model = model.cuda()
# with get_accelerator().device():
#     profile = get_model_profile(kron_resnet, input_shape=(1, 3, 224, 224), )
calcu_params(model)

            

start = time.time()
for i in range(1):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimzier.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        from utils.loss import s_l1
        loss += s_l1(model, 0.1)
        loss.backward()
        optimzier.step()

end = time.time()
print(f"kronecker model")
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
