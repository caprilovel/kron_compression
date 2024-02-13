import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    for param in model.parameters():
        total_params += param.numel()  # 统计参数总数
        sparse_params += torch.sum(torch.abs(param) < threshold).item()  # 统计绝对值小于阈值的参数数量

    sparsity = sparse_params / total_params  # 计算稀疏性
    return sparsity, sparse_params, total_params
    
