#import numpy as np
import tensorflow as tf
from keras import backend as K

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def evm_loss(y_true, y_pred):
    evm = K.sum(K.square(y_pred-y_true), axis=[3,2])
    return K.mean(evm, axis=1)

def sqrt_evm_loss(y_true, y_pred):
    evm = K.sum((K.square(y_pred-y_true)), axis=[3,2])
    evm = K.sqrt(K.abs(evm))                           
    return K.mean(evm, axis=1)
def cma_loss(y_true, y_pred):
    R2 = 1.
    loss = K.mean(K.square(K.sum(K.square(y_pred), axis=[3,2]) - K.square(R2)), axis=1)
    return loss
    
#a = [[[1.,2.,3.],[2.,3.,4.]],[[1.,2.,3.],[3.,4.,5.]]]
#b = [[[2.,3.,4.],[3.,4.,5.]],[[2.,3.,4.],[4.,5.,6.]]]
#a = tf.constant(a)
#b = tf.constant(b)
#a = tf.reshape(a, [2,2,3,1])
#b = tf.reshape(b, [2,2,3,1])

#sess = tf.Session()
#c = sess.run(cma_loss(a,b))
#print(c)

