
import similarity
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

X_test = np.load('test_set.npy')
Y_test = np.load('test_label.npy')
Z_test = np.load('test_snr.npy')
X_test_non_snr = np.load('test_set_non_snr.npy')
X_test_denoised_resnet = np.load('test_set_denoised.npy')

X_test_denoised_dwt = sio.loadmat('test_set_denoised_dwt.mat')
X_test_denoised_dwt = X_test_denoised_dwt['X']
X_test_denoised_dwt = X_test_denoised_dwt[0][0][0]

'''
S: similarity
0: (raw)-(non-snr) similarity
1: (wavelet_denoise)-(non-snr) similarity
2: (ResNet_denoise)-(non-snr) similarity
'''
S_test = np.zeros([len(Y_test), 3])

for i in range(len(Y_test)):
    a = time.clock()
    print(i)
    sample = X_test[i]
    sample_h1 = X_test_denoised_dwt[i]
    sample_h2 = X_test_denoised_resnet[i]
    sample_n = X_test_non_snr[i]
    
    S_test[i][0] = (similarity.hausdorff_distance(sample[0, 0:128], sample_n[0, 0:128])    
                    + similarity.hausdorff_distance(sample[1, 0:128], sample_n[1, 0:128]))/2
    S_test[i][1] = (similarity.hausdorff_distance(sample_h1[0, 0:128], sample_n[0, 0:128])    
                    + similarity.hausdorff_distance(sample_h1[1, 0:128], sample_n[1, 0:128]))/2   
    S_test[i][2] = (similarity.hausdorff_distance(sample_h2[0, 0:128], sample_n[0, 0:128])    
                    + similarity.hausdorff_distance(sample_h2[1, 0:128], sample_n[1, 0:128]))/2        
    print('time: %f'%(time.clock()-a))
    
np.save('similarity.npy', S_test)
sio.savemat('similarity.mat', {'S': S_test})
