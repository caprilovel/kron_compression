import torch
import torch.nn as nn
import torch.nn.functional as F
import time 

from models.KronLinear import KronLinear

class KronRegression(nn.Module):
    def __init__(self, input_dim, output_dim, kron_config=None):
        super(KronRegression, self).__init__()
        self.kronlinear = KronLinear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        
        return self.kronlinear(x)
    
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        
        return self.linear(x)

    
dataset = torch.randn(100, 10)
target = torch.randn(100, 1)

model = KronRegression(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
start_time = time.time()
# calculate backward flops
for i in range(100):
    output = model(dataset)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
end_time = time.time()   
print(f"training time: {end_time - start_time}")
 

model = LinearRegression(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
start_time = time.time()
# calculate backward flops
for i in range(100):
    output = model(dataset)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
end_time = time.time()
print(f"training time: {end_time - start_time}")



    
    

