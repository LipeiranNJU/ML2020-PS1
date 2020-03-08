import numpy as np
from math import pow
r = np.array([3.2, 3.8, 1.2, 4, 2.8])
k = 5
N = 5
tx2 = (12 * N) / (k * (k + 1)) * (np.dot(r, r) - k * pow(k + 1, 2) / 4)
tf = ((N - 1) * tx2) / (N * (k - 1) - tx2)
print(tx2)
print(tf)
CD005 = (2.728 * pow(k*(k+1)/6/N, 0.5))
print(CD005)
