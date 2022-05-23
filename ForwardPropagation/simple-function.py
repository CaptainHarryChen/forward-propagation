import random
import numpy as np
import matplotlib.pyplot as plt
import ForwardModel
from ForwardTensor import FTensor
from Optimizer import SGD, Adam


class SimpleFunction(ForwardModel.Model):
    def __init__(self, x=0., y=0.):
        super().__init__()
        self.x = FTensor(x)
        self.y = FTensor(y)
        self.register_param(self.x, "x")
        self.register_param(self.y, "y")

    def forward(self):
        return (1.5-self.x+self.x*self.y)**2+(2.25-self.x+self.x*self.y*self.y)**2+(2.625-self.x+self.x*self.y*self.y*self.y)**2

    def loss(self):
        return self.forward()


total_epochs = 100
propagation_times = 1
lr = 0.001

if __name__ == "__main__":
    x, y = 2.45, 1.23
    model = SimpleFunction(x, y)

    epoch_idx = np.arange(1, total_epochs+1)
    losses = np.zeros(total_epochs, dtype=np.float32)
    opti = SGD(lr=lr)

    model.train()
    for epoch in range(total_epochs):
        model.zero_grads()
        loss = model()
        # print(loss)
        losses[epoch] = loss
        opti.update(model.variables, model.grads)
        print(f"Epoch {epoch} :")
        # print(f"x = {x}  y = {y}")
        print(f"f(x,y) = {loss}")

    model.eval()
    print(f"Test: {model()}")

    plt.title("Forward Propagation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_idx, losses)
    plt.show()
