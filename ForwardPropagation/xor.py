import random
import numpy as np
import matplotlib.pyplot as plt
import ForwardModel
import Function as F
from ForwardTensor import FTensor
from Optimizer import SGD, Adam


class XOR(ForwardModel.Model):
    def __init__(self, hidden1):
        super().__init__()
        self.w1 = FTensor(np.random.standard_normal([hidden1, 4]))
        self.register_param(self.w1, "w1")
        self.b1 = FTensor(np.random.standard_normal([hidden1, 1]))
        self.register_param(self.b1, "b1")
        self.w2 = FTensor(np.random.standard_normal([1, hidden1]))
        self.register_param(self.w2, "w2")
        self.b2 = FTensor(np.random.standard_normal([1, 1]))
        self.register_param(self.b2, "b2")

    def forward(self, x):
        a1 = F.relu(FTensor.matmul(self.w1, x)+self.b1)
        y = F.sigmoid(FTensor.matmul(self.w2, a1)+self.b2)
        return y

    def loss(self, x, Y):
        return F.cross_entropy(self.forward(x), Y).flatten()


total_epochs = 250
propagation_times = 10
lr = 0.01

if __name__ == "__main__":
    dataX = np.zeros([16, 4, 1], dtype=np.int16)
    dataY = np.zeros([16, 1, 1], dtype=np.int16)
    for i in range(16):
        x = i
        for j in range(4):
            dataX[i][j][0] = x & 1
            dataY[i][0][0] ^= x & 1
            x >>= 1

    model = XOR(hidden1=128)

    epoch_idx = np.arange(1, total_epochs+1)
    losses = np.zeros(total_epochs, dtype=np.float32)
    opti = Adam(lr=lr)

    model.train()
    for epoch in range(total_epochs):
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
    for y_out, y_data in zip(output.flat, dataY.flat):
        predict = 1 if y_out > 0.5 else 0
        if predict == y_data:
            cnt += 1
    print(f"Accuracy: {cnt}/16  {(cnt/16.0):2.3f}%")
