'''

'''
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import similarity
import pywt 
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD
import wtLayer
import scipy.io as sio


#X_train = np.load('train_set.npy')
#Y_train = np.load('train_label.npy')
#Z_train = np.load('train_snr.npy')
#X_train_non_snr = np.load('train_set_non_snr.npy')

X_train = np.load('train_set4denoising.npy')
Y_train = np.load('train_label4denoising.npy')
Z_train = np.load('train_snr4denoising.npy')
X_train_non_snr = np.load('train_set_non_snr4denoising.npy')

X_test = np.load('test_set.npy')
Y_test = np.load('test_label.npy')
Z_test = np.load('test_snr.npy')
X_test_non_snr = np.load('test_set_non_snr.npy')

#X_test_denoised_dwt = sio.loadmat('test_set_denoised_dwt.mat')
#X_test_denoised_dwt = X_test_denoised_dwt['X']
#X_test_denoised_dwt = X_test_denoised_dwt[0][0][0]

classes = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']

def conv_block2(x, nb_filter = [16, 16], kernel_size = (1,4)):
    k1, k2 = nb_filter
    
    #1
    out = Convolution2D(k1, kernel_size=kernel_size, strides=1, 
                        data_format='channels_last', padding='same',
                        activation=None, init='glorot_uniform')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    
    #2
    out = Convolution2D(k2, kernel_size=kernel_size, strides=1, 
                        data_format='channels_last', padding='same',
                        activation=None, init='glorot_uniform')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out) 
    #3
    out = merge([out,x],mode='sum')
       
    
    return out

def conv_block1(x, nb_filter = [16, 16], kernel_size = (1,4)):
    k1, k2 = nb_filter
    
    out = Convolution2D(k1, kernel_size=kernel_size, strides=1, 
                        data_format='channels_last', padding='same',
                        activation=None, init='glorot_uniform')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    
    out = Convolution2D(k2, kernel_size=kernel_size, strides=1, 
                        data_format='channels_last', padding='same',
                        activation=None, init='glorot_uniform')(out)
    out = BatchNormalization()(out)
    
    x = Convolution2D(k2, kernel_size=kernel_size, strides=1, 
                        data_format='channels_last', padding='same',
                        activation=None, init='glorot_uniform')(x)
    x = BatchNormalization()(x)
    
    
    out = merge([out,x],mode='sum')
    out = Activation('relu')(out)    
    
    return out

#def dwt(inputs, wave = 'db6'):

    #in_phase = inputs[0]
    #quad = inputs[1]
    
    #coef_i = pywt.dwt(in_phase, wave)
    #coef_q = pywt.dwt(quad, wave)
    
    #detail = np.array([coef_i[1], coef_q[1]])
    #approx = np.array([coef_i[0], coef_q[0]])
    
    #return approx, detail

#def idwt(approx, detail, wave = 'db6'):
    #in_phase = pywt.idwt(tuple(approx[0]), tuple(detail[0]), wave, axis=0)
    #quad = pywt.idwt(tuple(approx[1]), tuple(detail[1]), wave, axis=0)
    #outputs = np.array([[in_phase, quad]])
    #return outputs

def training_model():
    #init
    inp = Input([2, 512])
    out = Reshape([2, 512, 1])(inp)
    out_1 = out
    #part1
    out = Convolution2D(16, kernel_size=(1, 4), strides=1, 
                        data_format='channels_last', padding='same',
                        activation=None, init='glorot_uniform')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    
    #part2
    out = conv_block2(out, [16, 16])
    out = conv_block2(out, [16, 16])
    out = conv_block2(out, [16, 16])
    out = conv_block2(out, [16, 16])
    out = conv_block2(out, [16, 16])
    out = conv_block2(out, [16, 16])
    #out = conv_block2(out, [16, 16])
    #out = conv_block2(out, [16, 16])
    #out = conv_block2(out, [16, 16])
    #out = conv_block2(out, [16, 16])
    #out = conv_block2(out, [16, 16])
    #out = conv_block2(out, [16, 16])
    
    #part3
    out = Convolution2D(1, kernel_size=(1, 4), strides=1, 
                               data_format='channels_last', padding='same',
                        activation=None, init='glorot_uniform')(out)
    out = BatchNormalization()(out)
    
    #part4
    out = merge([out,out_1], mode='sum')
    
    #last
    out = Reshape([2, 512])(out)
    model = Model(inp, out)
    return model



def training_part():
    in_shap = list(X_train.shape[1:]) 
    print(in_shap)
    
    model = training_model()
    '''
    def loss_func(y_true, y_pred):
        out = np.linalg.norm(y_pred - y_true)
        return out
    '''
    model.compile(loss='mse',
                      optimizer=SGD(lr=0.01, momentum=0.1, decay=0.001, nesterov=False),
                      metrics=['accuracy'])
    history = model.fit(X_train[0:10000], X_train_non_snr[0:10000],
                  epochs=25,
                  batch_size=256,
                  verbose=2,
                  #validation_data=None)
                  validation_data=(X_train[10010:12500], X_train_non_snr[10010:12500]))  
    with open('history.txt','w') as f:
        f.write(str(history.history))    
    model.save_weights('model_weights_ResNet4denoising.h5')
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    
def predict_part():
    
    
    model = training_model()
    model.load_weights('model_weights_ResNet4denoising.h5')
    
    num = 5
    sample = X_test[num]
    sample_hat = model.predict(np.array([sample]))
    
    sample_hat = model.predict(sample_hat)#
    sample_hat = model.predict(sample_hat)#
    sample_hat = model.predict(sample_hat)#
  
    
    sample_hat2 = X_test_denoised_dwt[num]
    
    sample_th = X_test_non_snr[num]
    
    print(sample_hat.shape)
    print(Y_test[num])
    print(Z_test[num])
    

    
    print('Frechet/Hausdorff distance:')
    print('fig. 1 and fig. 4 : %f'%(similarity.hausdorff_distance(sample[0], sample_th[0])))
    print('fig. 2 and fig. 4 : %f'%(similarity.hausdorff_distance(sample_hat2[0], sample_th[0])))
    print('fig. 3 and fig. 4 : %f'%(similarity.hausdorff_distance(sample_hat[0][0], sample_th[0])))
    
    #print('Frechet distance:')
    #print('fig. 1 and fig. 3 : %f'%(similarity.frechet_distance(sample[0], sample_th[0])))
    #print('fig. 2 and fig. 3 : %f'%(similarity.frechet_distance(sample_hat[0][0], sample_th[0])))    
    
    plt.figure(1)
    plt.plot(sample[0, 0:128])
    
    plt.figure(2)
    plt.plot(sample_hat2[0][0:128])    
    
    plt.figure(3)
    plt.plot(sample_hat[0][0, 0:128])
    
    plt.figure(4)
    plt.plot(sample_th[0, 0:128]) 
    
    plt.show()
    
def batch_denoising():
    model = training_model()
    model.load_weights('denoising2.h5')
    #X_train_denoisd = model.predict(X_train)
    #X_train_denoisd = model.prediAct(X_train_denoisd)
    #X_train_denoisd = model.predict(X_train_denoisd)
    #print(X_train_denoisd.shape)
    X_test_denoisd = model.predict(X_test)
    X_test_denoisd = model.predict(X_test_denoisd)
    #np.save('train_set_denoised.npy', X_train_denoisd)
    np.save('test_set_denoised.npy', X_test_denoisd)
training_part()
#predict_part()
#batch_denoising()

#from keras.utils import plot_model
#model = training_model()
#plot_model(model, to_file='model.png')
