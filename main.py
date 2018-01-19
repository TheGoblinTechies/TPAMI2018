# -*- coding: utf-8 -*-
import numpy as np
import copy
from keras.models import Model, Sequential
from keras.layers import Dense, Reshape, Flatten, Activation, Input, Lambda, merge
from keras.optimizers import Adam, RMSprop
from keras.layers.convolutional import Convolution2D
from keras.applications.vgg16 import VGG16
from keras import regularizers
from keras import backend as K
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from AEclass import autoencoder16
import math
import scipy.io as sio


if __name__=="__main__":
    learning_rate = 0.0001
    dim_feature = 16
    batchsize = 16
    alpha = 0
    beta = 1
    gamma = 1
    epoch_pretrain = 5
    epoch_train = 3
    epoch_alloctrain = 2
    epoch_dp = 1
    num_AE = 2
    temp_image = 50000
### test para
    total_image = 50000
    train_image = total_image
    test_image = 10000

    temp_batchsize = 8


    x_train = np.load('feat16_train.npy')
    #x_train += 4
    x_train /= 5
    x_test = np.load('feat16_test.npy')
    x_test /= 5
    #x_test /= 8


    #pretrain
    L = []

    for i in range(0,num_AE):
        opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
        firstAE = autoencoder16(learning_rate, dim_feature, batchsize, alpha, beta, gamma, identity = 1)
        firstAE.model.compile(loss=firstAE.RELoss(),optimizer = opt)
        firstAE.model.fit(x_train, x_train, batch_size = batchsize, epochs = epoch_pretrain)
        L.append(firstAE)



    x_bit_dist = np.zeros((num_AE,dim_feature))
#### assign photo to each AE
#### train iteration
    for turn in range (0,epoch_train):
        #initial
        min_Loss = 10000
        min_autoencoder = -1
        x_train_dist = []
        for num in range (0,num_AE):
            x_train_temp = np.zeros((1,dim_feature))
            x_train_dist.append(x_train_temp)

        #eg num_AE * 16 (0 or 1)
        x_bit_sum  = np.zeros((num_AE,dim_feature))
        x_bit_dist = np.zeros((num_AE,dim_feature))

        assign_image = []
        assign_AE = [0]*num_AE

        total_temp = x_train[:,:].reshape(total_image,dim_feature)

        total_pred = np.zeros((num_AE,total_image,dim_feature))
        for k in range (0,num_AE):
            total_pred[k] = L[k].model.predict(total_temp)

        for i in range (0,temp_image):
            min_Loss = 1e10
            min_autoencoder = -1
            temp = x_train[i,:].reshape(1,dim_feature)

            for k in range (0,num_AE):
                RELoss = np.sum(np.square( total_pred[k,i] - temp))
                #assign the feature to AE
                if min_Loss > RELoss:
                    min_autoencoder = k
                    min_Loss = RELoss


            #sum the RELoss of bits to each AE
            '''
            for j in range( 0 , dim_feature):
                for k in range(0,num_AE):
                    temp_bit_vector = np.square( L[k].model.predict(temp) - temp)
                    t = np.reshape(temp_bit_vector,(1,dim_feature))
                    x_bit_sum[k:k+1,0:dim_feature] += t
            '''

            assign_image.append(min_autoencoder)
            assign_AE[min_autoencoder]+=1

            x_train_dist[min_autoencoder] = np.append(x_train_dist[min_autoencoder], temp , axis=0)

        #找到重构误差之和最小的两个AE

        for feat in range(0 , dim_feature):
            final_first = 0
            final_second = 0
            min_loss = 1e10
            for first in range(0,num_AE-1):
                for second in range(first+1,num_AE):
                    loss_temp = 0
                    for image in range(0,temp_image):
                        loss_temp += min(np.square(total_pred[first,image,feat] - x_train[image,feat]),\
                                         np.square(total_pred[second,image,feat] - x_train[image,feat]))
                    if (min_loss > loss_temp):
                        min_loss = loss_temp
                        final_first = first
                        final_second = second
            print 'first, second',final_first, final_second

            x_bit_dist[final_first,feat] = 1
            x_bit_dist[final_second,feat] = 1


        #sum the RELoss of bits to each AE
        '''
        for j in range( 0 , dim_feature):
            min_bit_Loss = 1e10
            min_bit_AE = -1
            for k in range(0, num_AE):
                value = x_bit_sum[k,j]
                if min_bit_Loss > value:
                    min_bit_Loss = value
                    min_bit_AE = k
            x_bit_dist[min_bit_AE,j] = 1
        '''

        print "x_bit_dist"
        print x_bit_dist[0:num_AE,0:dim_feature]

        print "assign_AE", assign_AE
        print "x_bit_dist.shape", x_bit_dist.shape

        for k in range (0,num_AE):
            if assign_AE[k] != 0:
                #L[k].model.compile(loss=L[k].BitwiseRELoss(x_bit_dist[k,:],alpha),optimizer = opt)
                L[k].model.compile(loss=L[k].RELoss(),optimizer = opt)
                L[k].model.fit(x_train_dist[k][1:assign_AE[k]+1], x_train_dist[k][1:assign_AE[k]+1], batch_size = temp_batchsize, epochs = epoch_alloctrain)
        #L[0].model.compile(loss=L[0].BitwiseRELoss(x_bit_dist[0,:],alpha),optimizer = opt)
        #L[0].model.fit(x_train_0[1:temp_image+1], x_train_0[1:temp_image+1], batch_size = temp_batchsize, epochs = epoch_pretrain)

        assign_image = []
        assign_AE = [0]*num_AE

    #for k in range(0, num_AE):
    #    L[k].save('auto%dWeight.h5' % k)




    ### output
    train_AE_list = []
    test_AE_list = []

    train_pred = np.zeros((num_AE,train_image,dim_feature))
    train_encode = np.zeros((train_image,dim_feature))

    test_pred = np.zeros((num_AE,test_image,dim_feature))
    test_encode = np.zeros((test_image,dim_feature))



    for k in range(0,num_AE):
        train_pred[k] = L[k].model.predict(x_train)
        test_pred[k] = L[k].model.predict(x_test)

	sio.savemat('feat16_trn_real.mat', {'prob_train':train_pred[0]})
	sio.savemat('feat16_tst_real.mat', {'prob_test':test_pred[0]})

    for image in range (0,train_image):
        for j in range (0,dim_feature):
            docu_train = []
            for k in range(0, num_AE):
                if x_bit_dist[k, j] == 1:
                    docu_train.append(k)
            #print "docu 0, image, feature(j)", docu[0],image, j
            if np.abs(train_pred[docu_train[0],image, j]-x_train[image,j]) < np.abs(train_pred[docu_train[1],image, j]-x_train[image,j]):
                train_encode[image, j] = 1
        if image % 10000 == 0:
            print image/10000
        if image % 10000 == 0:
            print x_bit_dist
            print x_train[image,:]
            print train_pred[:,image,:]
            print train_encode[image]

    for image in range (0,test_image):
        for j in range (0,dim_feature):
            docu_test = []
            for k in range(0, num_AE):
                if x_bit_dist[k, j] == 1:
                    docu_test.append(k)
            #print "docu 0, image, feature(j)", docu[0],image, j
            if np.abs(test_pred[docu_test[0],image, j]-x_test[image,j]) < np.abs(test_pred[docu_test[1],image, j]-x_test[image,j]):
                test_encode[image, j] = 1
        if image % 1000 == 0:
            print image/10000
        if image % 1000 == 0:
            print x_bit_dist
            print x_test[image,:]
            print test_pred[:,image,:]
            print test_encode[image]

    np.save("feat16_trn_bi.npy", train_encode)
    np.save("feat16_tst_bi.npy", test_encode)

    x = np.load('feat16_trn_bi.npy')
    y = np.load('feat16_tst_bi.npy')

    sio.savemat('feat16_trn_bi.mat',{'prob_train':x})
    sio.savemat('feat16_tst_bi.mat',{'prob_test':y})

#	sio.savemat('feat16_trn_bi.mat', {'prob_train':train_encode})
#	sio.savemat('feat16_tst_bi.mat', {'prob_test':test_encode})

'''
        if image % 100 == 0:
            print x_bit_dist
            print x_test[image,:]
            print test_pred[:,image,:]
            print test_encode[image]
'''



'''
#### dp iteration
    for turn in range (0,epoch_dp):
        Loss1 = 0
        for image in range (0, temp_image):
            image_loss = 0

            total_code = 0

            for k in range (0, num_AE):
                total_C = 0
                for i in range (0,dim_feature):
                    total_C += x_bit_dist[k,i]
                image_loss += (total_C - np.sum(x_bit_dist[image,:,:])/num_AE )**2
                print "total_C",total_C,"average",np.sum(x_bit_dist[image,:,:])/num_AE
                total_code += total_C

            Loss1 += beta * image_loss + gamma * math.ceil(math.log(total_code,2)) - math.log(total_code)
            print image, Loss1

'''


####
