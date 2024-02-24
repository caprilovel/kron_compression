import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


def train(model, train_loader, optimizer, criterion, device, epoch, logger):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print every 1000 mini-batches
        if batch_idx % 1000 == 999:
            print(f'[{epoch + 1}, {batch_idx + 1}] loss: {running_loss / 1000}')
            logger.log_scalar('loss', running_loss / 1000, epoch * len(train_loader) + batch_idx)
            running_loss = 0.0
    return running_loss