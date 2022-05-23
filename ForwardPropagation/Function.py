import numpy as np
from ForwardTensor import FTensor


def relu(a):
    ret = FTensor((np.abs(a.value)+a.value)/2, a.delta)
    ret.delta = ret.delta*np.sign(ret.value)
    return ret


def log(a):
    return FTensor(np.log(a.value), a.delta/a.value)


def exp(a):
    return FTensor(np.exp(a.value), np.exp(a.value)*a.delta)


def sigmoid(a):
    return 1/(1+exp(-a))


def cross_entropy(a, y):
    return -(log(a)*y+log(1-a)*(1-y))


def average(a):
    return FTensor(np.average(a.value), np.average(a.delta))


def softmax(x, axis=-1):
    e_x = exp(x)
    probs = e_x / np.sum(e_x.value, axis=axis, keepdims=True)
    return probs
