import tensorflow as tf
from tensorflow.keras.models import  Sequential
import matplotlib.pyplot as plt

import numpy as np
from  tensorflow.keras.layers import Dense

x = np.linspace(0,100,30)
y = 2*x + 20 + 10 * np.random.randn(30)

plt.scatter(x,y)
plt.show()

model = Sequential()  #建立一个Sequential模型
model.add(Dense(1,input_dim=1))   #定义1个层的神经网络，输入参数和输出参数个数都是1
model.compile(optimizer='adam',loss='mse')  #定义优化器为adam算法和损失函数为均方误差
model.fit(x,y,epochs=8000)   #训练数据，训练5000次

model.save('lR.h5')
plt.scatter(x,y,c='r')
plt.plot(x,model.predict(x))
plt.show()




