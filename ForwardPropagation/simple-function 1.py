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
    debug_test()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x = np.arange(-10,10,0.5)
    y = np.arange(-10,10,0.5)
    x,y = np.meshgrid(x, y)
    z = forward(x,y)

    ax.plot_surface(x,y,z)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    ax.set_title("3D surface plot")
    plt.show()
