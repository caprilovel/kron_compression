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
