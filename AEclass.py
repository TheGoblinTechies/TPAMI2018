from keras.layers import Input, Dense
from keras.layers import Activation
from keras.models import Model
import numpy as np
import keras.backend as K
import tensorflow as tf
'''
def reconstruction_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
def stack_loss(x_feature,y_autoencoder):
    def distribution_loss(y_true, y_pred):
'''

class autoencoder16:
    def __init__(self,learning_rate, dim_feature, batchsize, alpha, beta, gamma, identity):
        # NOT USEFUL SO FAR
        self.learning_rate = 0.0001
        self.dim_feature = 16
        self.batchsize = 16
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.model= self._build_model()
        self.relax = 0.5
        self.id = identity

    def RELoss(self):
        def reconstruction_loss(y_true, y_pred):
            return K.sum(K.square(y_pred-y_true),axis= -1)
        return reconstruction_loss


    def BitwiseRELoss(self, x_bit_dist ,alpha):
        def BitwiseLoss(y_true, y_pred):

            output_list = []
            temp_pred = []
            temp_true = []
            num_post = [1]*self.dim_feature

            for i in range(self.dim_feature):
                temp_pred.append(y_pred[:,i])
                temp_true.append(y_true[:,i])

            p_true = tf.transpose(tf.stack(temp_true))

            for i in range(self.dim_feature):
                if x_bit_dist[i] == 0:
                    temp_pred[i] = y_true[:,i]

            output_list = temp_pred
            p_pred = tf.transpose(tf.stack(output_list))

            return K.sum(K.square(y_pred-y_true),axis= -1)+alpha*K.sum(K.square(p_true - p_pred))
        return BitwiseLoss

    def _build_model(self):
        input_feature = Input(shape=(16,))
        encoded = Dense(14,name='encode14',activation = 'tanh')(input_feature)
        encoded = Dense(12,name='encode12',activation = 'tanh')(encoded)
        encoded = Dense(10,name='encode10',activation = 'tanh')(encoded)

        encoded = Dense(8,name='middle8', activation = 'tanh')(encoded)

        decoded = Dense(10,name='decode10',activation = 'tanh')(encoded)
        decoded = Dense(12,name='decode12',activation = 'tanh')(decoded)
        decoded = Dense(14,name='decode14',activation = 'tanh')(decoded)
        decoded = Dense(16,name='decode16',activation = 'tanh')(decoded)

        model_autoencoder = Model(inputs= input_feature , outputs=decoded)
        return model_autoencoder

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)
########################################################3

'''
    def BitwiseRELoss(self, x_bit_dist ,alpha):
        def BitwiseLoss(y_true, y_pred):

'''



'''
    def train(self,X,xx,yy):
        opt=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.TOLoss(xx,yy),optimizer=opt)
        self.model.train_on_batch(X, X)

    def fit(self,X,xx,yy,epochs):
        opt=keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.TOLoss(xx,yy,X),optimizer=opt)
        #print('shape',X.shape)
        self.model.fit(X, X, batch_size=self.batchsize, epochs=epochs)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
'''
