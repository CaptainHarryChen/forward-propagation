import time
import numpy as np
import torch

def numpyTest():
    A = np.random.standard_normal([100,100,32,32])
    B = np.random.standard_normal([100,100,32,32])

    t1 = time.time()
    C = np.matmul(A,B)
    print(f"Time used (numpy): {time.time()-t1}s")


def torchTest():
    A = torch.randn([100,100,32,32])
    B = torch.randn([100,100,32,32])

    t1 = time.time()
    C = torch.matmul(A, B)
    print(f"Time used (torch): {time.time()-t1}s")

numpyTest()
torchTest()

numpyTest()
torchTest()