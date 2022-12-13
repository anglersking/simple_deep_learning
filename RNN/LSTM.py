import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Dense,LSTM,GRU
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

#导入手写数字数据集
(x_train,y_train),(x_test,y_test)  = mnist.load_data()

#数据归一化
x_train = x_train/255.   # 像素的值是0-255，归一化后变为0-1区间
x_test = x_test/255.

#对标签数据集转换为独热编码
y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)

cell_size = 100
input_size = 28
time_stemp = 28
model = Sequential()

model.add(
    LSTM(units=cell_size,input_shape=(time_stemp,input_size))
)

model.add(
    Dense(10,activation='softmax')
)

adam = Adam(lr=1e-4)
early_stopping = EarlyStopping(monitor = 'accuracy', patience = 50, verbose = 2)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=64,epochs=6,callbacks =[early_stopping])

i = np.random.randint(0,10000)

plt.imshow(x_test[i],cmap='gray')
plt.show()

img = x_test[i].reshape(1,28,28)
print(model.predict_classes(img))