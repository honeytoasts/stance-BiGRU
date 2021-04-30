# 3rd-party module
import torch.nn as nn
import torch.nn.functional as F

def loss_function(predict, target):

    # get cross entropy loss
    loss = F.cross_entropy(predict, target)

    return loss