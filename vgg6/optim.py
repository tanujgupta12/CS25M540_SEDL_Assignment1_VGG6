# Created by Tanuj Gupta (CS25M540)
"""Optimizer factory. Each optimizer gets a sensible default learning rate when none is given."""
import torch.optim as optim

# Adaptive methods (Adam, RMSprop, Nadam) diverge at SGD-sized learning rates,
# so each optimizer has its own default.
DEFAULT_LR = {
    "sgd": 0.05,
    "nesterov": 0.05,
    "adam": 1e-3,
    "adagrad": 1e-2,
    "rmsprop": 1e-3,
    "nadam": 1e-3,
}


def build_optimizer(name, params, lr=None, momentum=0.9, weight_decay=5e-4):
    lr = lr if lr is not None else DEFAULT_LR[name]
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nesterov":
        return optim.SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(DEFAULT_LR)}")
