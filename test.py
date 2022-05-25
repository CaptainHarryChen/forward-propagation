import torch
import time
import numpy as np

X = torch.tensor(np.random.standard_normal([128,128]),dtype=torch.float32)
Y = torch.tensor(np.random.standard_normal([128,128]),dtype=torch.float32)
tt1 = time.time()
Z = torch.nn.functional.linear(X, Y)
print(time.time()-tt1)
