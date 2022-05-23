import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class XOR(nn.Module):
    def __init__(self, hidden1):
        super().__init__()
        self.linear1 = nn.Linear(4, hidden1)
        self.linear2 = nn.Linear(hidden1, 1)

    def forward(self, x):
        a1 = F.relu(self.linear1(x))
        y = torch.sigmoid(self.linear2(a1))
        return y
    
    def loss(self,a,Y):
        return F.binary_cross_entropy(a, Y)

total_epochs = 500
propagation_times = 10
lr = 0.01

if __name__ == "__main__":
    dataX = np.zeros([16, 4], dtype=np.int16)
    dataY = np.zeros([16, 1], dtype=np.int16)
    for i in range(16):
        x = i
        for j in range(4):
            dataX[i][j] = x & 1
            dataY[i][0] ^= x & 1
            x >>= 1
        
    dataX = torch.tensor(dataX,dtype=torch.float32)
    dataY = torch.tensor(dataY,dtype=torch.float32)

    model = XOR(hidden1=128)

    epoch_idx = np.arange(1, total_epochs+1)
    losses = np.zeros(total_epochs, dtype=np.float32)
    opti = torch.optim.Adam(model.parameters(),lr=lr)

    model.train()
    for epoch in range(total_epochs):
        pred = model(dataX)
        loss = model.loss(pred,dataY)
        # print(loss)
        losses[epoch] = loss
        opti.zero_grad()
        loss.backward()
        opti.step()
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
    print(f"Accuracy: {cnt}/16  {(cnt/16.0)*100.0:2.3f}%")
