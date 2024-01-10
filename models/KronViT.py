# import math
# from collections import OrderedDict
# from functools import partial
# from typing import Any, Callable, Dict, List, NamedTuple, Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from utils.factorize import factorize
# from utils.decomposition import kron

# class MLP(torch.nn.Sequential):
#     """This block implements the multi-layer perceptron (MLP) module.

#     Args:
#         in_channels (int): Number of channels of the input
#         hidden_channels (List[int]): List of the hidden channel dimensions
#         norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
#         activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
#         inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
#             Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
#         bias (bool): Whether to use bias in the linear layer. Default ``True``
#         dropout (float): The probability for the dropout layer. Default: 0.0
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: List[int],
#         norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
#         inplace: Optional[bool] = None,
#         bias: bool = True,
#         dropout: float = 0.0,
#     ):
#         # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
#         # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
#         params = {} if inplace is None else {"inplace": inplace}

#         layers = []
#         in_dim = in_channels
#         for hidden_dim in hidden_channels[:-1]:
#             layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
#             if norm_layer is not None:
#                 layers.append(norm_layer(hidden_dim))
#             layers.append(activation_layer(**params))
#             layers.append(torch.nn.Dropout(dropout, **params))
#             in_dim = hidden_dim

#         layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
#         layers.append(torch.nn.Dropout(dropout, **params))

#         super().__init__(*layers)

# class ConvStemConfig(NamedTuple):
#     out_channels: int
#     kernel_size: int
#     stride: int
#     norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
#     activation_layer: Callable[..., nn.Module] = nn.ReLU

# class KronMLP(nn.Module):
#     def __init__(self, 
#                 in_channels: int,
#                 hidden_channels: List[int],
#                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
#                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
#                 inplace: Optional[bool] = None,
#                 bias: bool = True,
#                 dropout: float = 0.0,
#                 rank=10
#                 ):
#         params = {} if inplace is None else {"inplace": inplace}
        
#         in_a, in_b = factorize(in_channels)
#         hidden_a, hidden_b = factorize(hidden_channels)
#         self.a1 = nn.Parameter(torch.randn(rank, in_a, hidden_a))
#         self.b1 = nn.Parameter(torch.randn(rank, in_b, hidden_b))
#         nn.init.xavier_uniform_(self.a1)
#         nn.init.xavier_uniform_(self.b1)

#         self.a2 = nn.Parameter(torch.randn(rank, hidden_a, in_a))
#         self.b2 = nn.Parameter(torch.randn(rank, hidden_b, in_b))        
#         nn.init.xavier_uniform_(self.a2)
#         nn.init.xavier_uniform_(self.b2)

#         if norm_layer is not None:
#             self.norm = norm_layer(in_channels)
#         self.activation = activation_layer(**params)
#         self.dropout = nn.Dropout(dropout, **params)
        
#     def forward(self, x):
#         x = kron(self.a1, self.b1, x)
#         x = kron(self.a2, self.b2, x)
#         if hasattr(self, 'norm'):
#             x = self.norm(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         return x
    
# if __name__ == "__main__":
#     kronmlp = KronMLP(64, 128)
#     a = torch.randn(1,3,64)
#     kronmlp(a).shape         


# class KronMLPBlock(nn.Module):
#     """Transformer MLP block."""

#     _version = 2

#     def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
#         super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)
        
        
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.normal_(m.bias, std=1e-6)

#     def _load_from_state_dict(
#         self,
#         state_dict,
#         prefix,
#         local_metadata,
#         strict,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#     ):
#         version = local_metadata.get("version", None)

#         if version is None or version < 2:
#             # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
#             for i in range(2):
#                 for type in ["weight", "bias"]:
#                     old_key = f"{prefix}linear_{i+1}.{type}"
#                     new_key = f"{prefix}{3*i}.{type}"
#                     if old_key in state_dict:
#                         state_dict[new_key] = state_dict.pop(old_key)

#         super()._load_from_state_dict(
#             state_dict,
#             prefix,
#             local_metadata,
#             strict,
#             missing_keys,
#             unexpected_keys,
#             error_msgs,
#         )

# if __name__ == "__main__":
#     kronmlp = KronMLP(64, 128)
#     a = torch.randn(1,3,64)
    

# # class EncoderBlock(nn.Module):
# #     """Transformer encoder block."""

# #     def __init__(
# #         self,
# #         num_heads: int,
# #         hidden_dim: int,
# #         mlp_dim: int,
# #         dropout: float,
# #         attention_dropout: float,
# #         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
# #     ):
# #         super().__init__()
# #         self.num_heads = num_heads

# #         # Attention block
# #         self.ln_1 = norm_layer(hidden_dim)
# #         self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
# #         self.dropout = nn.Dropout(dropout)

# #         # MLP block
# #         self.ln_2 = norm_layer(hidden_dim)
# #         self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

# #     def forward(self, input: torch.Tensor):
# #         torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
# #         x = self.ln_1(input)
# #         x, _ = self.self_attention(x, x, x, need_weights=False)
# #         x = self.dropout(x)
# #         x = x + input

# #         y = self.ln_2(x)
# #         y = self.mlp(y)
# #         return x + y


# # class Encoder(nn.Module):
# #     """Transformer Model Encoder for sequence to sequence translation."""

# #     def __init__(
# #         self,
# #         seq_length: int,
# #         num_layers: int,
# #         num_heads: int,
# #         hidden_dim: int,
# #         mlp_dim: int,
# #         dropout: float,
# #         attention_dropout: float,
# #         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
# #     ):
# #         super().__init__()
# #         # Note that batch_size is on the first dim because
# #         # we have batch_first=True in nn.MultiAttention() by default
# #         self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
# #         self.dropout = nn.Dropout(dropout)
# #         layers: OrderedDict[str, nn.Module] = OrderedDict()
# #         for i in range(num_layers):
# #             layers[f"encoder_layer_{i}"] = EncoderBlock(
# #                 num_heads,
# #                 hidden_dim,
# #                 mlp_dim,
# #                 dropout,
# #                 attention_dropout,
# #                 norm_layer,
# #             )
# #         self.layers = nn.Sequential(layers)
# #         self.ln = norm_layer(hidden_dim)

# #     def forward(self, input: torch.Tensor):
# #         torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
# #         input = input + self.pos_embedding
# #         return self.ln(self.layers(self.dropout(input)))


# # class VisionTransformer(nn.Module):
# #     """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

# #     def __init__(
# #         self,
# #         image_size: int,
# #         patch_size: int,
# #         num_layers: int,
# #         num_heads: int,
# #         hidden_dim: int,
# #         mlp_dim: int,
# #         dropout: float = 0.0,
# #         attention_dropout: float = 0.0,
# #         num_classes: int = 1000,
# #         representation_size: Optional[int] = None,
# #         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
# #         conv_stem_configs: Optional[List[ConvStemConfig]] = None,
# #     ):
# #         super().__init__()
# #         _log_api_usage_once(self)
# #         torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
# #         self.image_size = image_size
# #         self.patch_size = patch_size
# #         self.hidden_dim = hidden_dim
# #         self.mlp_dim = mlp_dim
# #         self.attention_dropout = attention_dropout
# #         self.dropout = dropout
# #         self.num_classes = num_classes
# #         self.representation_size = representation_size
# #         self.norm_layer = norm_layer

# #         if conv_stem_configs is not None:
# #             # As per https://arxiv.org/abs/2106.14881
# #             seq_proj = nn.Sequential()
# #             prev_channels = 3
# #             for i, conv_stem_layer_config in enumerate(conv_stem_configs):
# #                 seq_proj.add_module(
# #                     f"conv_bn_relu_{i}",
# #                     Conv2dNormActivation(
# #                         in_channels=prev_channels,
# #                         out_channels=conv_stem_layer_config.out_channels,
# #                         kernel_size=conv_stem_layer_config.kernel_size,
# #                         stride=conv_stem_layer_config.stride,
# #                         norm_layer=conv_stem_layer_config.norm_layer,
# #                         activation_layer=conv_stem_layer_config.activation_layer,
# #                     ),
# #                 )
# #                 prev_channels = conv_stem_layer_config.out_channels
# #             seq_proj.add_module(
# #                 "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
# #             )
# #             self.conv_proj: nn.Module = seq_proj
# #         else:
# #             self.conv_proj = nn.Conv2d(
# #                 in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
# #             )

# #         seq_length = (image_size // patch_size) ** 2

# #         # Add a class token
# #         self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
# #         seq_length += 1

# #         self.encoder = Encoder(
# #             seq_length,
# #             num_layers,
# #             num_heads,
# #             hidden_dim,
# #             mlp_dim,
# #             dropout,
# #             attention_dropout,
# #             norm_layer,
# #         )
# #         self.seq_length = seq_length

# #         heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
# #         if representation_size is None:
# #             heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
# #         else:
# #             heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
# #             heads_layers["act"] = nn.Tanh()
# #             heads_layers["head"] = nn.Linear(representation_size, num_classes)

# #         self.heads = nn.Sequential(heads_layers)

# #         if isinstance(self.conv_proj, nn.Conv2d):
# #             # Init the patchify stem
# #             fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
# #             nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
# #             if self.conv_proj.bias is not None:
# #                 nn.init.zeros_(self.conv_proj.bias)
# #         elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
# #             # Init the last 1x1 conv of the conv stem
# #             nn.init.normal_(
# #                 self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
# #             )
# #             if self.conv_proj.conv_last.bias is not None:
# #                 nn.init.zeros_(self.conv_proj.conv_last.bias)

# #         if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
# #             fan_in = self.heads.pre_logits.in_features
# #             nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
# #             nn.init.zeros_(self.heads.pre_logits.bias)

# #         if isinstance(self.heads.head, nn.Linear):
# #             nn.init.zeros_(self.heads.head.weight)
# #             nn.init.zeros_(self.heads.head.bias)

# #     def _process_input(self, x: torch.Tensor) -> torch.Tensor:
# #         n, c, h, w = x.shape
# #         p = self.patch_size
# #         torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
# #         torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
# #         n_h = h // p
# #         n_w = w // p

# #         # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
# #         x = self.conv_proj(x)
# #         # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
# #         x = x.reshape(n, self.hidden_dim, n_h * n_w)

# #         # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
# #         # The self attention layer expects inputs in the format (N, S, E)
# #         # where S is the source sequence length, N is the batch size, E is the
# #         # embedding dimension
# #         x = x.permute(0, 2, 1)

# #         return x

# #     def forward(self, x: torch.Tensor):
# #         # Reshape and permute the input tensor
# #         x = self._process_input(x)
# #         n = x.shape[0]

# #         # Expand the class token to the full batch
# #         batch_class_token = self.class_token.expand(n, -1, -1)
# #         x = torch.cat([batch_class_token, x], dim=1)

# #         x = self.encoder(x)

# #         # Classifier "token" as used by standard language architectures
# #         x = x[:, 0]

# #         x = self.heads(x)

# #         return x


# # def _vision_transformer(
# #     patch_size: int,
# #     num_layers: int,
# #     num_heads: int,
# #     hidden_dim: int,
# #     mlp_dim: int,
# #     weights: Optional[WeightsEnum],
# #     progress: bool,
# #     **kwargs: Any,
# # ) -> VisionTransformer:
# #     if weights is not None:
# #         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
# #         assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
# #         _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
# #     image_size = kwargs.pop("image_size", 224)

# #     model = VisionTransformer(
# #         image_size=image_size,
# #         patch_size=patch_size,
# #         num_layers=num_layers,
# #         num_heads=num_heads,
# #         hidden_dim=hidden_dim,
# #         mlp_dim=mlp_dim,
# #         **kwargs,
# #     )

# #     if weights:
# #         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

# #     return model


# # _COMMON_META: Dict[str, Any] = {
# #     "categories": _IMAGENET_CATEGORIES,
# # }

# # _COMMON_SWAG_META = {
# #     **_COMMON_META,
# #     "recipe": "https://github.com/facebookresearch/SWAG",
# #     "license": "https://github.com/facebookresearch/SWAG/blob/main/LICENSE",
# # }


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
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
                FeedForward(dim, mlp_dim, dropout = dropout)
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
     