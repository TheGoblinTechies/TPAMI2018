import scipy.io as sio
import numpy as np
import pickle

x_train = np.load('feat16_train.npy')
x_test = np.load('feat16_test.npy')
sio.savemat('trn.mat', {'prob_train':x_train})
sio.savemat('tst.mat', {'prob_test':x_test})

print x_train.shape
