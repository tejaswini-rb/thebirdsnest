import torch
import torch.nn as nn
import torch.optim as optim

def gradient_based_pruning(module, amount=0.3):
    for name, param in module.named_parameters():
        if 'weight' in name:
            with torch.no_grad():
                if param.grad is not None:
                    # the pruning criterion based solely on gradient magnitudes
                    criterion = torch.abs(param.grad)
                    # cutoff threshold
                    threshold = torch.quantile(criterion, amount)
                    # create mask based on the gradient criterion and apply it
                    mask = criterion > threshold
                    param.data *= mask.float()