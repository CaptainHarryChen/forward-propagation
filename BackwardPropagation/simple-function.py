import torch
import numpy as np
import matplotlib.pyplot as plt


class SimpleFunction(torch.nn.Module):
    def __init__(self, x=0., y=0.):
        super().__init__()
        self.x = torch.nn.Parameter(torch.Tensor([x]))
        self.register_parameter("param_x", self.x)
        self.y = torch.nn.Parameter(torch.Tensor([y]))
        self.register_parameter("param_y", self.y)

    def forward(self):
        return (1.5-self.x+self.x*self.y)**2+(2.25-self.x+self.x*self.y*self.y)**2+(2.625-self.x+self.x*self.y*self.y*self.y)**2


total_epochs = 100
propagation_times = 5
lr = 0.001

if __name__ == "__main__":
    x, y = 2.45, 1.23
    model = SimpleFunction(x, y)

    epoch_idx = np.arange(1, total_epochs+1)
    losses = np.zeros(total_epochs, dtype=np.float32)
    opti = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    for epoch in range(total_epochs):
        loss = model()
        # print(loss)
        losses[epoch] = loss
        opti.zero_grad()
        loss.backward()
        opti.step()
        print(f"Epoch {epoch} :")
        # print(f"x = {x}  y = {y}")
        print(f"f(x,y) = {loss[0]}")

    model.eval()
    res = model()
    print(f"Test: {res[0]}")

    plt.title("Forward Propagation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_idx, losses)
    plt.show()
