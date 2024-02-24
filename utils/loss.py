import torch
import torch.nn as nn
import torch.nn.functional as F

from models.KronLinear import KronLinear

def s_l1(model, gamma=0.01):
    loss = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            loss += s_l1(module, 1)
        elif isinstance(module, KronLinear) and module.s is not None:
            loss += torch.norm(module.s, p=1)
    return gamma * loss

