import random
import numpy as np
import matplotlib.pyplot as plt
from ForwardTensor import FTensor


def forward(x, y):
    return (1.5-x+x*y)**2+(2.25-x+x*y*y)**2+(2.625-x+x*y*y*y)**2


def debug_test():
    def f(x, y):
        return 3*x**2+y
    x, y = 3, 4

    dx = FTensor(x)
    dy = FTensor(y)
    res = f(dx, dy)
    print(f"dx = {dx}  dy = {dy}")
    print(f"res= {res}")


total_epochs = 100
propagation_times = 5
lr = 0.001

if __name__ == "__main__":
    # debug_test()

    x, y = 2.45, 1.23
    print(f"x = {x}  y = {y}")
    print(f"f(x,y) = {forward(x,y)}")

    epoch_idx = np.arange(1, total_epochs+1)
    losses = np.zeros(total_epochs, dtype=np.float32)

    for epoch in range(total_epochs):
        delta = np.zeros(2)
        for prop_time in range(propagation_times):
            v = FTensor([x, y])
            res = forward(v[0], v[1])
            delta += v.delta * res.delta
            # print(res.delta, end=" ")
        # print(f"\n{delta}")
        delta /= propagation_times
        # print(delta)
        x -= delta[0] * lr
        y -= delta[1] * lr
        loss = forward(x, y)
        losses[epoch] = loss
        print(f"Epoch {epoch} :")
        print(f"x = {x}  y = {y}")
        print(f"f(x,y) = {loss}")

    plt.title("Forward Propagation Loss")
    plt.xlabel("Epoch id")
    plt.ylabel("Loss")
    plt.plot(epoch_idx, losses)
    plt.show()
