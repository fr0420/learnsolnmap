from torch import nn


ACTIVATION_DICT = {
    'ELU': nn.ELU,
    'Hardshrink': nn.Hardshrink,
    'Hardsigmoid': nn.Hardsigmoid,
    'Hardtanh': nn.Hardtanh,
    'Hardswish': nn.Hardswish,
    'LeakyReLU': nn.LeakyReLU,
    'LogSigmoid': nn.LogSigmoid,
    'MultiheadAttention': nn.MultiheadAttention,
    'PReLU': nn.PReLU,
    'ReLU': nn.ReLU,
    'ReLU6': nn.ReLU6,
    'RReLU': nn.RReLU,
    'SELU': nn.SELU,
    'CELU': nn.CELU,
    'GELU': nn.GELU,
    'Sigmoid': nn.Sigmoid,
    'SiLU': nn.SiLU,
    'Softplus': nn.Softplus,
    'Softshrink': nn.Softshrink,
    'Softsign': nn.Softsign,
    'Tanh': nn.Tanh,
    'Tanhshrink': nn.Tanhshrink,
    'Threshold': nn.Threshold
}


def get_activation(activation_fn, **kwargs):
    """Define activation function"""
    
    return ACTIVATION_DICT[activation_fn](**kwargs)
