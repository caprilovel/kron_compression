from typing import List
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import time

def factorize(n: int) -> List[int]:
    """Return the most average two factorization of n."""
    for i in range(int(np.sqrt(n)) + 1, 1, -1):
        if n % i == 0:
            return [i, n // i]
    return [n, 1]

class KronLinear(nn.Module):
    def __init__(self, input_dim, output_dim, rank=0, structured_sparse=False, bias=True) -> None:
        """Kronecker Linear Layer

        the weight matrix is a kron(a, b) matrix
        
        Args:
            rank (int): the rank of the Kronecker product
            a_shape (tuple): the shape of the **a** matrix 
            b_shape (tuple): the shape of the **b** matrix
            structured_sparse (bool, optional): _description_. Defaults to False.
            bias (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        input_dims = factorize(input_dim)
        output_dims = factorize(output_dim)

        if rank == 0:
            rank = min(*input_dims, *output_dims) // 2 + 1
        self.rank = rank
        
        a_shape = [input_dims[0], output_dims[1]]
        b_shape = [input_dims[1], output_dims[0]]
        

        self.structured_sparse = structured_sparse
        if structured_sparse:
            self.s = nn.Parameter(torch.randn( *a_shape), requires_grad=True)
        self.a = nn.Parameter(torch.randn(rank, *a_shape), requires_grad=True)
        self.b = nn.Parameter(torch.randn(rank, *b_shape), requires_grad=True)
        self.a_shape = self.a.shape
        self.b_shape = self.b.shape

        
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)
        bias_shape = np.multiply(a_shape, b_shape)
        if bias:
            self.bias = nn.Parameter(torch.randn(*bias_shape[1:]), requires_grad=True)
        else:
            self.bias = None
        
    def forward(self, x):
        start_time = time.time()
        
        a = self.a
        
        if self.structured_sparse:
            a = self.s.unsqueeze(0) * self.a
        
        # a = self.s.unsqueeze(0) * self.a
        # w = kron(a, self.b)
        x_shape = x.shape 
        b = self.b
        
        x = torch.reshape(x, (-1, x_shape[-1]))
        
        temp_time1 = time.time()
        # b = rearrange(b, 'r b1 b2 -> b1 (b2 r)')
        b = b.permute(1, 2, 0).contiguous().view(b.shape[1], -1).contiguous()
        
        # x = rearrange(x, 'n (a1 b1) -> n a1 b1', a1=self.a_shape[1], b1=self.b_shape[1])
        x = x.view(-1, self.a_shape[1], self.b_shape[1])
        
        
        out = x @ b
        
        # out = rearrange(out, 'n a1 (b2 r) -> r (n b2) a1', b2=self.b_shape[2], r=self.rank) 
        out = out.view(-1, self.a_shape[1], self.rank, self.b_shape[2])

        # Permute dimensions
        out = out.permute(2, 0, 3, 1)
        out = out.contiguous().view(self.rank, -1, self.a_shape[1])
        temp_time2 = time.time()
        # for i in range(self.rank):
            
        
        out = torch.bmm(out, a)
        out = torch.sum(out, dim=0).squeeze(0)
        
        
        # out = rearrange(out, '(n b2) a2 -> n (a2 b2)', b2=self.b_shape[2])
        out = out.view(-1, self.b_shape[2], self.a_shape[2])

        # Permute dimensions
        out = out.permute(0, 2, 1).contiguous()

        # Reshape again
        out = out.view(-1, self.a_shape[2] * self.b_shape[2])
        
        
        
        out = torch.reshape(out, x_shape[:-1] + (self.a_shape[2] * self.b_shape[2],))
        
        temp_time3 = time.time()
        
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        tt = time.time()
        print(f"temp1: {temp_time1 - start_time}, temp2: {temp_time2 - temp_time1}, temp3: {temp_time3 - temp_time2}")
        print(f"{tt - start_time}")
        return out
    
L = nn.Linear(5000, 5000)
KL = KronLinear(5000, 5000, structured_sparse=False)

input = torch.randn(256, 100, 5000)
L, KL = L.to('cuda'), KL.to('cuda')
input = input.to('cuda')


L_time = time.time()
L(input)
L_time = time.time() - L_time
print(f"L_time: {L_time}")
KL(input)


