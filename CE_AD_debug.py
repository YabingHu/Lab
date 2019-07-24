# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:06:01 2019

@author: yabinghu
"""
from scipy.io import loadmat
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt 
#%%
AD = loadmat('140000_195_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_train=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_train=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_train=np.vstack((kl,bl)).T
#y_train=bl
'''
#AD = loadmat('140000_190_250_400_650_100_100_AD.mat')
#AD = loadmat('110000_160_250_400_650_100_100_AD.mat')
AD = loadmat('120000_172_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_train2=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_train2=np.vstack((kl,bl)).T

x_train=np.vstack((x_train,x_train2))
y_train=np.vstack((y_train,y_train2))


#AD = loadmat('90000_160_250_400_650_100_100_AD.mat')
AD = loadmat('120000_172_250_400_650_50_50_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_train2=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_train2=np.vstack((kl,bl)).T
#y_test=bl
x_train=np.vstack((x_train,x_train2))
y_train=np.vstack((y_train,y_train2))

'''

AD = loadmat('90000_160_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_train2=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_train2=np.vstack((kl,bl)).T
#y_test=bl
x_train=np.vstack((x_train,x_train2))
y_train=np.vstack((y_train,y_train2))

AD = loadmat('110000_160_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_train2=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_train2=np.vstack((kl,bl)).T
#y_test=bl
x_train=np.vstack((x_train,x_train2))
y_train=np.vstack((y_train,y_train2))


#####################test data
AD = loadmat('90000_195_250_400_650_100_100_AD.mat')
#AD = loadmat('140000_185_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T
#y_test=bl

m, n = x_train.shape  
model = Sequential()
model.add(Dense(100, input_dim=n, activation='relu',kernel_initializer='normal'))
#model.add(Dropout(0.25))
model.add(Dense(100, activation='relu',kernel_initializer='normal'))
#model.add(Dropout(0.25))
model.add(Dense(100, activation='relu',kernel_initializer='normal'))
#model.add(Dropout(0.25))
model.add(Dense(2,kernel_initializer='normal',activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mse'])
res=model.fit(x_train, y_train, epochs = 5,validation_data=(x_test,y_test),batch_size=128) 

plt.figure(0)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(res.history['val_mean_squared_error'],label='test error')
plt.plot(res.history['mean_squared_error'],label=' training error')
plt.legend(loc='best')
plt.title('learning curve for three layer neural network')

prediction = model.predict(x_test)
plt.figure(1)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 90000_190_250_400_650_100_100_AD.mat')

plt.figure(2)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 90000_190_250_400_650_100_100_AD.mat')

#%%
AD = loadmat('140000_190_250_400_650_100_100_AD.mat')
#AD = loadmat('110000_160_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T
#y_test=bl

prediction = model.predict(x_test)
plt.figure(3)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 190')

plt.figure(4)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 190')



AD = loadmat('140000_145_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T
#y_test=bl

prediction = model.predict(x_test)
plt.figure(5)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 145')

plt.figure(6)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 145')


AD = loadmat('90000_160_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T
#y_test=bl

prediction = model.predict(x_test)
plt.figure(5)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 90000_160')

plt.figure(6)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 90000_160')



AD = loadmat('105000_190_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T
#y_test=bl

prediction = model.predict(x_test)
plt.figure(5)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 105000_190')

plt.figure(6)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 105000_190')




AD = loadmat('130000_145_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T
#y_test=bl

prediction = model.predict(x_test)
plt.figure(5)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 130000_145')

plt.figure(6)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 130000_145')



AD = loadmat('90000_145_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T
#y_test=bl

prediction = model.predict(x_test)
plt.figure(5)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 90000_145')

plt.figure(6)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 90000_145')



AD = loadmat('100000_185_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T
#y_test=bl

prediction = model.predict(x_test)
plt.figure(5)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 100000_185')

plt.figure(6)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 100000_185')








AD = loadmat('120000_172_250_400_650_150_150_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T
#y_test=bl

prediction = model.predict(x_test)
plt.figure(5)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 120000_172_250_400_650_150_150_AD.mat')

plt.figure(6)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 120000_172_250_400_650_150_150_AD.mat')





AD = loadmat('120000_172_250_400_650_50_50_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T
#y_test=bl

prediction = model.predict(x_test)
plt.figure(5)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 120000_172_250_400_650_50_50_AD.mat')

plt.figure(6)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 120000_172_250_400_650_50_50_AD.mat')

#%%

########Combined test data

AD = loadmat('120000_172_250_400_650_50_50_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_train=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T
#y_train=bl

AD = loadmat('140000_190_250_400_650_100_100_AD.mat')
#AD = loadmat('90000_145_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test2=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test2=np.vstack((kl,bl)).T

x_test=np.vstack((x_test,x_test2))
y_test=np.vstack((y_test,y_test2))

prediction = model.predict(x_test)
plt.figure(5)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 100000_185+90000_145')

plt.figure(6)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 100000_185+90000_145')
#%%

#######################################################
AD = loadmat('140000_190_250_400_650_100_100_AD.mat')
#AD = loadmat('110000_160_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test=np.vstack((kl,bl)).T

AD = loadmat('100000_185_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test1=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_train=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test1=np.vstack((kl,bl)).T
#y_train=bl

#AD = loadmat('140000_190_250_400_650_100_100_AD.mat')
AD = loadmat('90000_145_250_400_650_100_100_AD.mat')
state=AD['x']#[xb,xb_dot,xf,xf_dot]
state=np.transpose(state)
xb=state[0]
xb_dot=state[1]
xf=state[2]
xf_dot=state[3]
kl=AD['kl']
bl=AD['bl']
kl=kl.reshape(len(kl))
bl=bl.reshape(len(bl))
x_test2=np.vstack((xb,xf,xb_dot,xf_dot,xf_dot**2,xb_dot*xf_dot)).T
#x_test=np.vstack((xb,xf,xb_dot,xf_dot,kl)).T
y_test2=np.vstack((kl,bl)).T

x_test0=np.vstack((x_test1,x_test2))
y_test0=np.vstack((y_test1,y_test2))

x_test=np.vstack((x_test,x_test0))
y_test=np.vstack((y_test,y_test0))

prediction = model.predict(x_test)
plt.figure(5)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,0],label='calculated kl')
plt.plot(y_test[:,0],label='true kl')
plt.legend(loc='best')
plt.title('calculated kl vs true kl for 140000_190+100000_185+90000_145')

plt.figure(6)
plt.grid(color='g', linestyle='-', linewidth=0.5)
plt.plot(prediction[:,1],label='calculated bl')
plt.plot(y_test[:,1],label='true bl')
plt.legend(loc='best')
plt.title('calculated bl vs true bl for 140000_190+100000_185+90000_145')