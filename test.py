import numpy as np

a = np.zeros((1, 4))
b = np.zeros((1, 1))

c = np.concatenate((a, b), axis=1)
print(c.shape)
