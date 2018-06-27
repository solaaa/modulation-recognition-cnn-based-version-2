import scipy.io as sio
import numpy as np


def one_hot(y, class_num):
    ret = np.zeros([y.shape[0], class_num])
    for i in range(len(y)):
        ret[i][int(y[i])-1] = 1.
    return ret

def complex_to_iq(data):
    num = len(data)
    length = len(data[0])
    ret = np.zeros([num, 2, length])
    ret[:, 0, :] = np.real(data)
    ret[:, 1, :] = np.imag(data)
    return ret

def get_train_data(start, end, path='D:\\matlab2016b\\workplace\\signal_channel\\dataset\\recognize_set\\with_channel\\channel1\\batch_'):
    assert start > 0
    assert start <= end
    class_num = 10
    data_path = path + 'data'
    data_label = path + 'label'
    data_snr = path + 'snr'    
    
    while(True):
        for i in range(start, end + 1):
            data = sio.loadmat(data_path+str(i)+'.mat')['dataset']
            label = sio.loadmat(data_label+str(i)+'.mat')['labelset']
            snr = sio.loadmat(data_snr+str(i)+'.mat')['snrset']
            label = one_hot(label, class_num)
            data = complex_to_iq(data)
            out_shape = list(data.shape)
            inp_shape = np.append(out_shape, 1)             
            data = data.reshape(inp_shape)
            
            yield(data, label)

def get_val_data(start, end, path='D:\\matlab2016b\\workplace\\signal_channel\\dataset\\recognize_set\\with_channel\\channel1_val\\batch_'):
    assert start > 0
    assert start <= end
    class_num = 10
    data_path = path + 'data'
    data_label = path + 'label'
    data_snr = path + 'snr'    
    
    while(True):
        for i in range(start, end + 1):
            data = sio.loadmat(data_path+str(i)+'.mat')['dataset']
            label = sio.loadmat(data_label+str(i)+'.mat')['labelset']
            snr = sio.loadmat(data_snr+str(i)+'.mat')['snrset']
            label = one_hot(label, class_num)
            data = complex_to_iq(data)
            out_shape = list(data.shape)
            inp_shape = np.append(out_shape, 1)            
            data = data.reshape(inp_shape)
            yield(data, label)

