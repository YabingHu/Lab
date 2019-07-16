# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:25:59 2019

@author: yabinghu
"""
from scipy.io import loadmat
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt 
from sklearn.preprocessing import normalize
AD = loadmat('110000_160_250_400_650_100_100_AD.mat')
#AD = loadmat('110000_190_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
bl_n=AD['bl_n'][0]
kl_n=AD['kl_n'][0]
data_y=-(kl_n*xf_dot)*(xb_dot*xf_dot)
normalized_data_y = normalize([data_y])
normalized_data_y=np.reshape(normalized_data_y, data_y.shape)
data_x=np.vstack((xb_dot,xf_dot)).T
x_train=data_x[:int(len(data_x)*0.9)]
y_train=data_y[:int(len(data_y)*0.9)]
x_test=data_x[int(len(data_x)*0.9):]
y_test=data_y[int(len(data_y)*0.9):]

m, n = x_train.shape  
model = Sequential()
model.add(Dense(100, input_dim=n, activation='relu',kernel_initializer='normal'))
model.add(Dense(100, activation='relu',kernel_initializer='normal'))
model.add(Dense(100, activation='relu',kernel_initializer='normal'))
model.add(Dense(1,kernel_initializer='normal',activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mse'])
#result_3=model.fit(x_train, y_train,epochs=5,batch_size=128)
res=model.fit(x_train, y_train, epochs = 10,validation_data=(x_test,y_test),batch_size=128) 
predictions = model.predict(x_test)

plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(res.history['val_mean_squared_error'],label='test error')
plt.plot(res.history['mean_squared_error'],label=' training error')
plt.legend(loc='best')
plt.title('learning curve for three layer neural network')

'''
plt.plot(1-np.asarray(result_1.history['acc']),'ro',label=' 1 hidden layer')
plt.plot(1-np.asarray(result_2.history['acc']),'bo',label=' 2 hidden layer')
plt.plot(1-np.asarray(result_3.history['acc']),'go',label=' 3 hidden layer')
plt.legend(loc='best')
plt.title('training error for three neural network')
plt.xlabel('number of epoch')
plt.ylabel('error')
'''
