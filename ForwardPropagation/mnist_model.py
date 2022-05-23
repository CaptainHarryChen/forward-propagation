import random
import time
import numpy as np
import matplotlib.pyplot as plt
import ForwardModel
import Layer
import Function as F
from ForwardTensor import FTensor
from Optimizer import SGD, Adam
import mnist_loader


class MNIST_model(ForwardModel.Model):
    def __init__(self, hidden1, hidden2):
        super().__init__()
        self.linear1 = Layer.Linear(784, hidden1, self, "linear1")
        self.linear2 = Layer.Linear(hidden1, hidden2, self, "linear2")
        self.linear3 = Layer.Linear(hidden2, 10, self, "linear3")

    def forward(self, x):
        a1 = F.relu(self.linear1(x))
        a2 = F.relu(self.linear2(a1))
        a3 = self.linear3(a2)
        y = F.softmax(a3, axis=-1)
        return y

    def loss(self, x, Y):
        return F.cross_entropy(self.forward(x), Y)

    def correct(self, a, Y):
        return Y[np.argmax(a)] == 1


images, labels = mnist_loader.load_data(".\\MNIST", "train")
train_dataX, train_dataY = mnist_loader.standardize(images, labels)
images, labels = mnist_loader.load_data(".\\MNIST", "t10k")
test_dataX, test_dataY = mnist_loader.standardize(images, labels)


total_epochs = 5
propagation_times = 1
lr = 0.01
mini_batch_size = 32

model = MNIST_model(hidden1=128, hidden2=128)

epoch_idx = np.arange(1, total_epochs+1)
losses = np.zeros(total_epochs, dtype=np.float32)
opti = Adam(lr=lr)

time_st = time.time()

model.train()
for epoch in range(total_epochs):
    idx = np.arange(len(train_dataX), dtype=np.int32)
    np.random.shuffle(idx)
    sum_loss = 0.0
    n = len(train_dataX)

    model.zero_grads()
    for j in range(0, n, mini_batch_size):
        dataX = train_dataX[idx[j:j + mini_batch_size]]
        dataY = train_dataY[idx[j:j + mini_batch_size]]
        loss = model(dataX, dataY)
        sum_loss += loss
        # print(f"loss: {loss}")
    opti.update(model.variables, model.grads)

    average_loss = sum_loss / (n//mini_batch_size)
    losses[epoch] = average_loss
    print(f"Epoch {epoch} :")
    # print(f"x = {x}  y = {y}")
    print(f"Average loss = {average_loss}")

time_ed = time.time()
print(f"Time used: {(time_ed-time_st):2.3f}s")

plt.title("Forward Propagation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(epoch_idx, losses)
plt.show()

model.eval()
output = model(test_dataX)
cnt = 0
for i in range(len(output)):
    if model.correct(output[i], test_dataY[i]):
        cnt += 1
print(
    f"Accuracy: {cnt}/{len(test_dataX)}  {(cnt/len(test_dataX)*100.0):2.3f}%")
