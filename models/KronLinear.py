import numpy as np
import torch 
import torch.nn as nn
from gkpd.tensorops import kron

class KronLinear(nn.Module):
    def __init__(self, rank, a_shape, b_shape, structured_sparse=False, bias=True) -> None:
        """Kronecker Linear Layer

        Args:
            rank (int): _description_
            a_shape (_type_): _description_
            b_shape (_type_): _description_
            structured_sparse (bool, optional): _description_. Defaults to False.
            bias (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.rank = rank
        self.structured_sparse = structured_sparse
        if structured_sparse:
            self.s = nn.Parameter(torch.randn( *a_shape), requires_grad=True)
        self.a = nn.Parameter(torch.randn(rank, *a_shape), requires_grad=True)
        self.b = nn.Parameter(torch.randn(rank, *b_shape), requires_grad=True)
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)
        bias_shape = np.multiply(a_shape, b_shape)
        if bias:
            self.bias = nn.Parameter(torch.randn(*bias_shape[1:]), requires_grad=True)
        else:
            self.bias = None
        
    def forward(self, x):
        a = self.a
        if self.structured_sparse:
            a = self.s.unsqueeze(0) * self.a
        
        # a = self.s.unsqueeze(0) * self.a
        w = kron(a, self.b)
        
        out = x @ w 
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out
    
