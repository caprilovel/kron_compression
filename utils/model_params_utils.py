import torch 
import torch.nn as nn
import torch.nn.functional as F

def count_parameter(model):
    total = 0
    for i in model.parameters():
        total += i.numel()
    return total

def sparse_count(model, thresold=1e-5):
    total = 0
    sparse = 0
    for i in model.parameters():
        total += i.numel()
        sparse += torch.sum(torch.abs(i) < thresold).item()
    return sparse, total



    