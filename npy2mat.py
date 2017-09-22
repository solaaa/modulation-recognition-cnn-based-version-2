import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
a = np.load('test_set.npy')
sio.savemat('test_set.mat', {'test_set': a})

#a = sio.loadmat('test_set_denoised_dwt.mat')
#print(a)
#X = a['X']
#print(X[0][0][0].shape)
#X = X[0][0][0]
#print(X)


