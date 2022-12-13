import tensorflow as tf
from  tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist  #导入手写数字识别的数据集，如果本地没有，会从官网下载
#(60000,28,28)         (10000,28,28)
(x_train, y_train) , (x_test,y_test) = mnist.load_data()

# print(x_train[0])
#
# print(y_train[0])

plt.imshow(x_train[1],cmap='gray')

plt.show()



#(60000,28,28) ---> (60000,784)
x_train = x_train.reshape(x_train.shape[0],-1)
#(10000,28,28) ---> (10000,784)
x_test = x_test.reshape(x_test.shape[0],-1)

#print(x_train.shape)

#print(y_train[2])

#把标签数据转换为独热（onehot)编码
y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)

#print(y_train[2])

model = Sequential()

model.add(
    Dense(units=10,       #输出的维度 10
          input_dim=784,  #输入维度 784
          bias_initializer = 'one',  # 神经网络节点权重偏移值
          activation='softmax',    #激活函数
          )
)

sgd = SGD(lr=0.2) #定义随机梯度下降优化算法，学习率为0.2

model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=['accuracy']
)

model.fit(x_train,y_train,batch_size=64,epochs=30)

loss,accuracy = model.evaluate(x_test,y_test)

print('loss=',loss)
print('accuracy=',accuracy)

model.save('singleNet.h5')



