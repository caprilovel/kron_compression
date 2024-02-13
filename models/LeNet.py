import torch 
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.LeakyReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.LeakyReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.relu5(self.fc3(x))
        return x
    
    def group_LASSO(self, alpha=0.01):
        l2_norm1 = torch.norm(self.fc2.weight, p=2, dim=0)
        l2_norm2 = torch.norm(self.fc3.weight, p=2, dim=0)
        l1_norm1 = torch.norm(torch.cat([l2_norm1, l2_norm2]), p=1)
        return l1_norm1 * alpha
    
class LeNet_5(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(25 * 4 * 4, 120)
        self.relu3 = nn.LeakyReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.LeakyReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 25 * 4 * 4)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.relu5(self.fc3(x))
        return x
    
    def group_LASSO(self, alpha=0.01):
        l2_norm1 = torch.norm(self.fc2.weight, p=2, dim=0)
        l2_norm2 = torch.norm(self.fc3.weight, p=2, dim=0)
        l1_norm1 = torch.norm(torch.cat([l2_norm1, l2_norm2]), p=1)
        return l1_norm1 * alpha
    
