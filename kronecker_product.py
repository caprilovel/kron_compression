#%%
import torch 
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision import transforms
#%%
trans = transforms.ToTensor()
cifar10_train = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=trans, download=True
)
cifar10_test = torchvision.datasets.CIFAR10(
    root='./data', train=False, transform=trans, download=True
)
#%%
def calculate_sparsity(model):
    total_params = 0
    zero_params = 0
    for name, parameter in model.named_parameters():
        if 'weight' in name:
            total_params += parameter.numel()
            zero_params += torch.sum(parameter == 0).item()
        
    sparsity = zero_params / total_params
    return sparsity

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(32 * 32 * 3, 1000)
        self.activate = nn.Sigmoid()
        self.layer2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activate(x)
        x = self.layer2(x)
        return x
    
def group_lasso(model, lambda_value=0.1):
    regularization_loss = 0
    for name, parameter in model.named_parameters():
        # 对属于同一组的参数进行L1正则化
        if 'weight' in name:
            regularization_loss += torch.norm(parameter, p=1)
    # 将正则化项添加到总损失中
    return lambda_value * regularization_loss

def LASSO(model, lambda_value=1):
    l1_reg = torch.tensor(0., requires_grad=True).to(device)
    for name, parameter in model.named_parameters():
        # 对属于同一组的参数进行L1正则化
        if 'weight' in name:
            l1_reg += torch.norm(parameter, p=1)
    # 将正则化项添加到总损失中
    return lambda_value * l1_reg

train_loader = data.DataLoader(cifar10_train, batch_size=5000, shuffle=True)
test_loader = data.DataLoader(cifar10_test, batch_size=5000, shuffle=False)
# %%
classes = cifar10_train.classes
cifar10_train.data.shape
#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SimpleModel()
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

#%%
for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.shape[0], -1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss += LASSO(model)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSparsity: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), calculate_sparsity(model)))
            print('group lasso loss: {:.6f}'.format(group_lasso(model).item()))


# %%
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc = nn.Linear(32 * 8 * 8, 10)  # 32是最后一个卷积层的输出通道数，8*8是图像大小经过两次池化的结果

    def forward(self, x):
        # 第一个卷积层
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第二个卷积层
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 展平数据
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)

        return x

# 创建模型实例
model = SimpleCNN()
model = model.cuda()
for epoch in range(100):
    for batch_idx, (d, t) in enumerate(train_loader):
        d, t = d.cuda(), t.cuda()
        optimizer.zero_grad()
        output = model(d)
        loss = criterion(output, t)
        loss += LASSO(model)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSparsity: {:.6f}'.format(
                epoch, batch_idx * len(d), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), calculate_sparsity(model)))
            print('group lasso loss: {:.6f}'.format(group_lasso(model).item()))

        
# %%
