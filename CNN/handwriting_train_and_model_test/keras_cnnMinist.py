import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.optimizers import Adam

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(-1,28,28,1)  #加颜色通道层 （60000,28,28)->(60000,28,28,1)
x_test = x_test.reshape(-1,28,28,1)   #加颜色通道层 （10000,28,28)->(10000,28,28,1)


y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)  #把标签改为独热（one hot）编码
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)


model = Sequential()

#添加第一个卷积层
model.add(
    Conv2D(
        input_shape=(28,28,1),  #输入的形状
        filters=32,             #卷积核的个数
        kernel_size=(5,5),      #卷积核大小
        strides= 1,             #卷积的步长
        padding= 'same',        #填充的模式
        activation= 'relu'      #激活函数
    )
)
model.layers[0].get_weights()
#添加了一个池化层
model.add(
    MaxPooling2D(
        pool_size=2,     #池化的大小
        strides= 2,
        padding='same'
    )
)

#添加第二个卷积层
model.add(
    Conv2D(
        filters=64,  # 卷积核的个数
        kernel_size=(5, 5),  # 卷积核大小
        strides=1,  # 卷积的步长
        padding='same',  # 填充的模式
        activation='relu'  # 激活函数
    )

)

model.add(
    MaxPooling2D(
        pool_size=2,  # 池化的大小
        strides=2,
        padding='same'
    )

)

#添加一个dropout层

model.add(
    Dropout(0.2)   #表示丢弃的个数 表示保留80%数据
)

#添加一个全连接层,首先把数据扁平化
model.add(
    Flatten()
)

model.add(
    Dense(1024,activation='relu')
)
#添加第二个个dropout层
model.add(
    Dropout(0.5)
)
#添加第二个全连接层

model.add(
    Dense(10,activation='softmax')
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  #使用交叉熵做做损失函数
    metrics=['accuracy']
)

model.fit(x_train,y_train,batch_size=128,epochs=1,validation_data=(x_test,y_test))

model.save('myCNNMnist.h5')


