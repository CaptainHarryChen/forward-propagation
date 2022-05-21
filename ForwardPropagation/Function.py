import numpy as np
from ForwardTensor import FTensor


def abs(a):
    ret=FTensor(a)
    ret.value[a.value<0]=-a.value[a.value<0]
    ret.delta[a.value<0]=-a.delta[a.value<0]
    return ret

def relu(a):
    ret=FTensor(a)
    ret.value[a.value<0]=0
    ret.delta[a.value<0]=0
    return ret

def log(a):
    return FTensor(np.log(a.value),a.delta/a.value)

def exp(a):
    return FTensor(np.exp(a.value),np.exp(a.value)*a.delta)

def sigmoid(a):
    return 1/(1+exp(-a))

def cross_entropy(a,y):
    return -(log(a)*y+log(1-a)*(1-y))

def average(a):
    return FTensor(np.average(a.value),np.average(a.delta))
    