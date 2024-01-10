import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定义简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 加载CIFAR-10数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=2)

# 初始化模型和优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

net = net.to(device)

# 定义LASSO正则化的权重衰减参数
lasso_lambda = 10
def calculate_sparsity(model):
    total_params = 0
    zero_params = 0
    for name, parameter in model.named_parameters():
        if 'weight' in name:
            total_params += parameter.numel()
            zero_params += torch.sum(parameter == 0).item()
        
    sparsity = zero_params / total_params
    return sparsity, zero_params, total_params
# 训练模型
for epoch in range(1000):  # 仅为演示目的，实际训练中可增加epoch数
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # 正向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # LASSO正则化
        l1_regularization = torch.tensor(0., requires_grad=True).to(device)
        for param in net.parameters():
            l1_regularization += torch.norm(param, 1)
        
        loss += lasso_lambda * l1_regularization

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每100个小批次打印一次损失
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            print('Sparsity: %.3f' % calculate_sparsity(net)[0], 'zero_nums: %d' % calculate_sparsity(net)[1], 'total_params: %d' % calculate_sparsity(net)[2])

print('Finished Training')
