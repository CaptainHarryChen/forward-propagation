import torch
import Function as F
from ForwardTensor import FTensor


class Model():
    def __init__(self, propagation_time=10):
        self.param = {}
        self.variables = {}
        self.grads = {}
        self.propagation_time = propagation_time
        self.mode = "train"

    def register_param(self, param: FTensor, name: str):
        self.param[name] = param
        self.variables[name] = param.value
        self.grads[name] = torch.zeros(param.delta.shape)

    def zero_grads(self):
        for key in self.grads.keys():
            self.grads[key].fill_(0)

    def reset_param(self):
        for key in self.param.keys():
            self.param[key].reset()

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"

    def setPropagationTime(new_propagation_time):
        self.propagation_time = new_propagation_time

    def forward(self, *argv, **kwargv):
        pass

    def loss(self, *argv, **kwargv):
        pass

    def __call__(self, *argv, **kwargv):
        if self.mode == "train":
            for i in range(self.propagation_time):
                self.reset_param()
                res = self.loss(*argv, **kwargv)
                res = F.average(res)
                for key in self.grads.keys():
                    delta_grads = self.param[key].delta * res.delta
                    self.grads[key] += delta_grads / self.propagation_time
            return res.value
        if self.mode == "eval":
            return self.forward(*argv, **kwargv).value
