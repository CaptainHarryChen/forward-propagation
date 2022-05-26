import torch
import ForwardModel
from ForwardTensor import FTensor


class Linear():
    def __init__(self, in_features, out_features, model, name):
        k = torch.sqrt(torch.tensor(1.0/in_features))
        self.w = FTensor(torch.zeros([out_features, in_features]).uniform_(-k, k))
        model.register_param(self.w, name+"_w")
        self.b = FTensor(torch.zeros([out_features, 1]).uniform_(-k, k))
        model.register_param(self.b, name+"_b")

    def __call__(self, x):
        x = FTensor.expand_dims(x, axis=-1)
        a = FTensor.matmul(self.w, x)+self.b
        a = FTensor.squeeze(a, axis=-1)
        return a
