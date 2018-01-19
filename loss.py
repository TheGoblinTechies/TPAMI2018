import numpy as np
import random

y_pred=np.zeros((4,16))
dim_feature = 16
x_bit_dist=np.ones((16))

for j in range(0,4):
    for i in range(0,16):
        y_pred[j,i] = random.random()

print y_pred



output_list = []
temp = []
num_post = [1]*dim_feature

for i in range(dim_feature):
    temp.append(y_pred[:,i])

p0 = np.transpose(np.stack(temp))
print 'p0',p0

for i in range(dim_feature):
    if x_bit_dist[i] == 0:
        temp[i] = y_pred[:,i] * 0

output_list = temp

p = np.transpose(np.stack(output_list))
print 'p',p

print np.sum(np.square(p-p0))
