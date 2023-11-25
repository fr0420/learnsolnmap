import torch
import math


def xavier_uniform_init(model, gain=1.0):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.endswith(".weight"):
            bound = gain * math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
            param.data.uniform_(-bound, bound)


def xavier_normal_init(model, gain=1.0):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.endswith(".weight"):
            std = gain * math.sqrt(2) / math.sqrt(param.shape[0] + param.shape[1])
            param.data.normal_(0, std)


def kaiming_uniform_init(model, gain=1.0):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.endswith(".weight"):
            if "layers.0" in name:  # The first layer does not have ReLU applied on its input
                bound = gain * math.sqrt(3) / math.sqrt(param.shape[1])
                param.data.uniform_(-bound, bound)
            else:
                bound = gain * math.sqrt(6) / math.sqrt(param.shape[1])
                param.data.uniform_(-bound, bound)


def kaiming_normal_init(model, gain=1.0):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.endswith(".weight"):
            if "layers.0" in name:  # The first layer does not have ReLU applied on its input
                param.data.normal_(0, gain * 1 / math.sqrt(param.shape[1]))
            else:
                param.data.normal_(0, gain * math.sqrt(2) / math.sqrt(param.shape[1]))
