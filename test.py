import time
from concurrent.futures import ThreadPoolExecutor

def fun(x):
    print(f"hahah{x}")
    return x+1

with ThreadPoolExecutor(max_workers=2) as executor:
    ans = executor.map(fun, [1,2,3,4])
    for res in ans:
        print(res)