import torch
import torch.nn as nn
import torch.optim as optim

def undecayed_pruning(module, epsilon=0.01, amount=0.9):
    for name, param in module.named_parameters():
        if 'weight' in name:
            with torch.no_grad():
                tensor = param.data
                grad = param.grad
                if grad is not None:
                    print("here3")
                    criterion = torch.abs(-tensor * grad + epsilon * tensor**2)
                    threshold = torch.quantile(criterion, amount)
                    mask = criterion > threshold
                    param.data = tensor * mask.float()