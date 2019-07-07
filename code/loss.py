import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

margin = 1
def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=False, reduction='elementwise_mean', pos_weight=None):

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)

    if pos_weight is None:
        ce_loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    else:
        log_weight = 1 + (pos_weight - 1) * target
        ce_loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
    
    #    if math.isnan(ce_loss.mean()):
    #       ce_loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp() + 0.1).log())
     #   print(input.grad)
    #    return torch.Tensor([0]) 
            
    if weight is not None:
        ce_loss = ce_loss * weight

    if reduction == False:
        return ce_loss
    elif reduction == 'elementwise_mean':
        return ce_loss.mean()
    else:
        return ce_loss.sum()

class Loss_classi(nn.Module):
    def __init__(self):
        super(Loss_classi, self).__init__()

    def loss_classi(self, output, label):

        pos = torch.sum(label)
        pos_num = F.relu(pos - 1) + 1
        total = torch.numel(label)
        neg_num = F.relu(total - pos - 1) + 1
        pos_w = neg_num / pos_num

        classi_loss = binary_cross_entropy_with_logits(output, label, pos_weight=pos_w, reduce=True)
        
        return classi_loss

    def forward(self, output, label):

        loss = self.loss_classi(output, label)
        return loss
