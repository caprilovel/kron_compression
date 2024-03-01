import torch
import torch.nn as nn
import torch.nn.functional as F
import time 



from deepspeed.profiling.flops_profiler import get_model_profile

from models.KronLinear import KronLinear, factorize
class KronRegression(nn.Module):
    def __init__(self, input_dim, output_dim, kron_config=None):
        super(KronRegression, self).__init__()
        self.kronlinear = KronLinear(input_dim, output_dim)
        
        in_shape = factorize(input_dim, 0)
        out_shape = factorize(output_dim, 0)
        
        self.a = nn.Parameter(torch.randn(in_shape[0], out_shape[0]), requires_grad=True) # (input_dim1, output_dim1)
        self.b = nn.Parameter(torch.randn(in_shape[1], out_shape[1]), requires_grad=True) # (input_dim2, output_dim2)
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        
    def forward(self, x):
        # x shape (batch, (input_dim1, input_dim2))
        x = x.view(-1, self.in_shape[-1]) # x (batch * input_dim1, input_dim2)
        out = x @ self.b # out (batch * input_dim1, output_dim2)
        out = out.view(-1, self.in_shape[0], self.out_shape[1]) # out (batch, input_dim1, output_dim2)
        out = out.permute(0, 2, 1) # out (batch, output_dim2, input_dim1)
        out = out.reshape(-1, self.in_shape[0]) # out (batch * output_dim2, input_dim1)
        out = out @ self.a # out (batch * output_dim2, output_dim1)
        out = out.view(-1, self.out_shape[1], self.out_shape[0]) # out (batch, output_dim2, output_dim1)
        out = out.permute(0, 2, 1) # out (batch, output_dim1, output_dim2)
        out = out.reshape(-1, self.output_dim) # out (batch, output_dim)
        
        return out

class KronRegression2(nn.Module):
    def __init__(self, input_dim, out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kronlinear = KronLinear(input_dim, out_dim)
        self.input_dim = input_dim
        self.out_dim = out_dim
        
    def forward(self, x):
        return self.kronlinear(x)
        

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x):
        
        return self.linear(x)
dimensions = [10000 * (i + 1) for i in range(20)]
Flops = []
for i in dimensions:
    flops, macs, params =  get_model_profile(KronRegression2(i, 1), (100, i))
    Flops.append(flops)



regression_flops = []
for i in dimensions:
    flops, macs, params =  get_model_profile(LinearRegression(i, 1), (100, i))
    regression_flops.append(flops)
print(Flops)
print(regression_flops)
import matplotlib.pyplot as plt
# Flops = ['20 K', '25 K', '30 K', '40 K', '40 K', '48 K', '50 K', '50 K', '60 K', '50 K']
# change '20 K' into number
def convert_flop(flop):
    if 'K' in flop:
        return int(flop.replace('K', '')) * 1000
    elif 'M' in flop:
        return int(flop.replace('M', '')) * 1000000
    return int(flop)
Flops = [convert_flop(flop) for flop in Flops]
regression_flops = [convert_flop(flop) for flop in regression_flops]
# y axis in 'K' or 1e-3 

plt.plot(dimensions, Flops, label='KronRegression')
plt.yticks([10000 * (i + 2) for i in range (7)], [str(i+2) + '0K' for i in range(7)])
plt.savefig('kron_regression_flops.png')

plt.clf()

plt.plot(dimensions, regression_flops, label='LinearRegression')
plt.yticks([5000000 * (i + 1) for i in range (8)], [str(i *5 + 5) + 'M' for i in range(8)])
plt.savefig('linear_regression_flops.png')
batch_size = 1
dataset = torch.randn(batch_size, 10)
target = torch.randn(batch_size, 1)

model = KronRegression(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
start_time = time.process_time()
# calculate backward flops
for i in range(100):
    output = model(dataset)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
end_time = time.process_time()   
print(f"training time: {end_time - start_time}")
 

model = LinearRegression(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
start_time = time.process_time()
# calculate backward flops
for i in range(100):
    output = model(dataset)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
end_time = time.process_time()
print(f"training time: {end_time - start_time}")



    
    

