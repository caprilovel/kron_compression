import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.KronLinear import KronLinear

def calculate_matrix_sparsity(matrix):
    total_elements = matrix.numel()
    zero_elements = torch.sum(matrix == 0).item()
    sparsity = zero_elements / total_elements
    return sparsity, zero_elements

def calculate_model_sparsity(model):
    total_elements = 0
    zero_elements = 0
    for param in model.parameters():
        total_elements += param.numel()
        zero_elements += torch.sum(param == 0).item()
    sparsity = zero_elements / total_elements
    return sparsity

def calculate_accuracy(predictions, targets):
    total_elements = predictions.numel()
    correct_elements = torch.sum(predictions == targets).item()
    accuracy = correct_elements / total_elements
    return accuracy

# calcu params of model
def calcu_params(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print(f"total params: {total_params}")
    return total_params

def calculate_sparsity(model, threshold=1e-6):
    total_params = 0
    sparse_params = 0

    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            sparsity, sparse_params, total_params = calculate_sparsity(module, threshold)
            total_params += total_params
            sparse_params += sparse_params
            
        elif isinstance(module, KronLinear):
            if module.s is not None:    
                total_params += module.s.numel()  # 统计参数总数
                sparse_params += torch.sum(torch.abs(module.s) < threshold).item()
                
    return sparse_params / total_params, sparse_params, total_params
