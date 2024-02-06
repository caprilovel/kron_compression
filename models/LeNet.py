import torch 
import torch.nn as nn
import torch.nn.functional as F


# class KronLeNet(nn.Module):
#     def __init__(self, group_id=1) -> None:
#         super(KronLeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
#         self.relu1 = nn.LeakyReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.relu2 = nn.LeakyReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         rank1 = 21
#         rank2 = 10
#         rank3 = 4
        
#         self.kronfc1 = KronLinear(rank1, Kronnecker_group[group_id][0][0], Kronnecker_group[group_id][0][1], bias=False, structured_sparse=True)
        
#         self.kronfc2 = KronLinear(rank2, Kronnecker_group[group_id][1][0], Kronnecker_group[group_id][1][1], bias=False, structured_sparse=True)
        
#         self.kronfc3 = KronLinear(rank3, Kronnecker_group[group_id][2][0], Kronnecker_group[group_id][2][1], bias=False, structured_sparse=True)
#         self.relu3 = nn.LeakyReLU()
#         self.relu4 = nn.LeakyReLU()


#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = self.relu3(self.kronfc1(x))
#         x = self.relu4(self.kronfc2(x))
#         x = self.kronfc3(x)
#         return x


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