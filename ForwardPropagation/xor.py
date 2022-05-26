import random
import torch
import matplotlib.pyplot as plt
import ForwardModel
import Layer
import Function as F
from ForwardTensor import FTensor
from Optimizer import SGD, Adam


class XOR(ForwardModel.Model):
    def __init__(self, hidden1):
        super().__init__()
        self.linear1 = Layer.Linear(4, hidden1, self, "linear1")
        self.linear2 = Layer.Linear(hidden1, 1, self, "linear2")

    def forward(self, x):
        a1 = F.relu(self.linear1(x))
        y = F.sigmoid(self.linear2(a1))
        return y

    def loss(self, x, Y):
        return F.cross_entropy(self.forward(x), Y).flatten()


total_epochs = 500
propagation_times = 1
lr = 0.01

if __name__ == "__main__":
    dataX = torch.zeros([16, 4], dtype=torch.int16)
    dataY = torch.zeros([16, 1], dtype=torch.int16)
    for i in range(16):
        x = i
        for j in range(4):
            dataX[i][j] = x & 1
            dataY[i][0] ^= x & 1
            x >>= 1
    dataX = dataX.float()
    dataY = dataY.float()

    model = XOR(hidden1=128)

    epoch_idx = torch.arange(1, total_epochs+1)
    losses = torch.zeros(total_epochs, dtype=torch.float32)
    opti = Adam(lr=lr)

    model.train()
    for epoch in range(total_epochs):
        model.zero_grads()
        loss = model(dataX, dataY)
        # print(loss)
        losses[epoch] = loss
        opti.update(model.variables, model.grads)
        print(f"Epoch {epoch} :")
        # print(f"x = {x}  y = {y}")
        print(f"loss = {loss}")

    plt.title("Forward Propagation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_idx, losses)
    plt.show()

    model.eval()
    output = model(dataX)
    # print(output)
    cnt = 0
    for y_out, y_data in zip(output.flatten(), dataY.flatten()):
        predict = 1 if y_out > 0.5 else 0
        if predict == y_data:
            cnt += 1
    print(f"Accuracy: {cnt}/16  {(cnt/16.0):2.3f}%")
