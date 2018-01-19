import scipy.io as sio
import numpy as np
x = np.load('feat16_trn_bi.npy')
y = np.load('feat16_tst_bi.npy')

sio.savemat('feat16_trn_bi.mat',{'prob_train':x})
sio.savemat('feat16_tst_bi.mat',{'prob_test':y})
