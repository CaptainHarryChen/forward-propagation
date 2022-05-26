import torch
import torch.nn.functional as F
from ForwardTensor import FTensor


def relu(a):
    ret = FTensor(F.relu(a.value), a.delta)
    ret.delta = ret.delta*torch.sign(ret.value)
    return ret


def log(a):
    return FTensor(torch.log(a.value), a.delta/a.value)


def exp(a):
    return FTensor(torch.exp(a.value), torch.exp(a.value)*a.delta)


def sigmoid(a):
    return 1/(1+exp(-a))


def cross_entropy(a, y):
    return -(log(a)*y+log(1-a)*(1-y))


def average(a):
    return FTensor(torch.mean(a.value), torch.mean(a.delta))


def softmax(x, axis=-1):
    e_x = exp(x)
    probs = e_x / torch.sum(e_x.value, axis=axis, keepdims=True)
    return probs
