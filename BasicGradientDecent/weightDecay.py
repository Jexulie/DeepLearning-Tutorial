import numpy as np

#! Pseudo Code ...

W = np.arange(0, 100)

penalty = 0
for i in np.arange(0, W.shape[0]):
    for j in np.arange(0, W.shape[1]):
        penalty += (W[i][j] ** 2)

