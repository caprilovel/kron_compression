import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from gkpd.tensorops import kron
# from utils.factorize import factorize
from typing import Optional
from timm.layers import DropPath
from einops import rearrange

from torch.jit import Final

from timm.layers import use_fused_attn
from typing import List
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
        a = self.a
        if self.structured_sparse:
            a = self.s.unsqueeze(0) * self.a
        
        # a = self.s.unsqueeze(0) * self.a
        # w = kron(a, self.b)
        x_shape = x.shape 
        b = self.b
        r = self.a_shape[0]
        x = torch.reshape(x, (-1, x_shape[-1]))
        b = rearrange(b, 'r b1 b2 -> b1 (b2 r)')
        x = rearrange(x, 'n (a1 b1) -> n a1 b1', a1=self.a_shape[1], b1=self.b_shape[1])
        out = x @ b
        out = rearrange(out, 'n a1 (b2 r) -> r (n b2) a1', b2=self.b_shape[2], r=r)
        out = torch.bmm(out, a)
        out = torch.sum(out, dim=0).squeeze(0)
        out = rearrange(out, '(n b2) a2 -> n (a2 b2)', b2=self.b_shape[2])
        out = torch.reshape(out, x_shape[:-1] + (self.b_shape[2],))
        
        
        
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out
    
def KronLinear2Linear(kronlinear):
    # turn a kronlinear layer into a linear layer
    weight = torch.kron(kronlinear.a, kronlinear.b)
    bias_flag = kronlinear.bias is not None
    linear = nn.Linear(weight.shape[0], weight.shape[1], bias=bias_flag)
    linear.weight.data = weight
    if kronlinear.bias is not None:
        linear.bias.data = kronlinear.bias
    return linear

class KronMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features


        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = KronLinear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = KronLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = KronMlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


    
    
    
    
