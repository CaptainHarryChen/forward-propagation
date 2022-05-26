import torch


class FTensor:

    def __init__(self, value, delta=None, dtype=torch.float32):
        if isinstance(value, FTensor):
            self.value = value.value.clone()
            self.delta = value.delta.clone()
        else:
            self.value = torch.as_tensor(value, dtype=dtype)
            if delta is not None:
                self.delta = torch.as_tensor(delta, dtype=dtype)
                assert(self.value.shape == self.delta.shape)
            else:
                self.delta = torch.randn(self.value.shape)
        self.shape = self.value.shape
        self.size = self.value.size

    def reset(self):
        self.delta = torch.randn(self.value.shape)

    def __add__(self, B):
        if isinstance(B, FTensor):
            return FTensor(self.value+B.value, self.delta+B.delta)
        return FTensor(self.value+B, self.delta)

    def __radd__(self, B):
        return self.__add__(B)

    def __sub__(self, B):
        if isinstance(B, FTensor):
            return FTensor(self.value-B.value, self.delta-B.delta)
        return FTensor(self.value-B, self.delta)

    def __rsub__(self, B):
        return FTensor(B-self.value, -self.delta)

    def __neg__(self):
        return FTensor(-self.value, -self.delta)

    def __mul__(self, B):
        if isinstance(B, FTensor):
            return FTensor(self.value*B.value, self.value*B.delta+self.delta*B.value)
        return FTensor(self.value*B, self.delta*B)

    def __rmul__(self, B):
        if isinstance(B, FTensor):
            return FTensor(B.value*self.value, B.delta*self.value+B.value*self.delta)
        return FTensor(B*self.value, B*self.delta)

    def __truediv__(self, B):
        if isinstance(B, FTensor):
            assert(B.value.size == 1)
            assert(B.delta.flatten()[0] == 0)
            B = B.value.flatten()[0]
        return FTensor(self.value/B, self.delta/B)

    def __rtruediv__(self, B):
        if isinstance(B, FTensor):
            assert(B.value.size == 1)
            assert(B.delta.flatten()[0] == 0)
            B = B.value.flatten()[0]
        return FTensor(B/self.value, -B/(self.value*self.value)*self.delta)

    def matmul(A, B):
        if isinstance(A, FTensor) and isinstance(B, FTensor):
            return FTensor(torch.matmul(A.value, B.value), torch.matmul(A.value, B.delta)+torch.matmul(A.delta, B.value))
        if isinstance(A, FTensor):
            return FTensor(torch.matmul(A.value, B), torch.matmul(A.delta, B))
        if isinstance(B, FTensor):
            return FTensor(torch.matmul(A, B.value), torch.matmul(A, B.delta))
        return torch.matmul(A, B)

    def __pow__(self, B):
        if B == 2:
            return self.__mul__(self)
        assert(B == 2)

    def __str__(self):
        return f"({self.value}, {self.delta})"

    def __getitem__(self, idx):
        return FTensor(self.value[idx], self.delta[idx])

    def flatten(self):
        return FTensor(self.value.flatten(), self.delta.flatten())

    def expand_dims(x, axis=-1):
        if isinstance(x, FTensor):
            return FTensor(torch.unsqueeze(x.value, axis), torch.unsqueeze(x.delta, axis))
        return torch.unsqueeze(x, axis)
    
    def squeeze(x, axis=-1):
        if isinstance(x, FTensor):
            return FTensor(torch.squeeze(x.value, axis), torch.squeeze(x.delta, axis))
        return torch.squeeze(x, axis)
