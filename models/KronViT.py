import torch
from torch import nn
import sys
sys.path.append("/home/zhu.3723/kron_compression/")
# print(sys.path)
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
from gkpd.gkpd import gkpd
from utils.factorize import factorize

from models.KronLinear import KronLinear


class KronMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dim_factor=0, hidden_dim_factor=0, rank=10):
        super().__init__()
        if dim_factor == 0:
            dim_factor = factorize(dim)
        self.dim_factor = dim_factor
        
        if hidden_dim_factor == 0:
            hidden_dim_factor = factorize(hidden_dim)
        self.hidden_dim_factor = hidden_dim_factor
        
        in_a, in_b = dim_factor
        out_a, out_b = hidden_dim_factor
        
        self.net = nn.Sequential(
            KronLinear(rank, (in_a, out_a), (in_b, out_b)),
            nn.GELU(),
            KronLinear(rank, (out_a, in_a), (out_b, in_b))
        )
        
    def forward(self, x):
        return self.net(x)

class KronFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., dim_factor=0, hidden_dim_factor=0, rank=10):
        super().__init__()
        if dim_factor == 0:
            dim_factor = factorize(dim)
        self.dim_factor = dim_factor
        
        if hidden_dim_factor == 0:
            hidden_dim_factor = factorize(hidden_dim)
        self.hidden_dim_factor = hidden_dim_factor
        
        in_a, in_b = dim_factor
        out_a, out_b = hidden_dim_factor
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            KronLinear(rank, (in_a, out_a), (in_b, out_b)),
            nn.GELU(),
            nn.Dropout(dropout),
            KronLinear(rank, (out_a, in_a), (out_b, in_b)),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
    
 

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                KronFeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
if __name__ == "__main__":
    vit = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    test_tensor = torch.randn(1, 3, 256, 256)
    print(vit(test_tensor).shape)
    # kronmlp = KronMLP(256, 512)
    # a = torch.randn(1, 256)
    # b = kronmlp(a)
    # print(b.shape)
     