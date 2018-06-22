# Python 3 version - PyTorch Implementation
"""Contains activation functions and gradient descent optimizers."""

import torch


activations = {
    "sigmoid": torch.nn.Sigmoid,
    "softmax": torch.nn.Softmax,
    "rectify": torch.nn.ReLU,
    "selu": torch.nn.SELU
}

optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam
}
