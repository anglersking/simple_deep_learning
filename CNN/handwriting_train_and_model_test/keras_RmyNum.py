import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# mnist = tf.keras.datasets.mnist
# (x_train, y_train) , (x_test,y_test) = mnist.load_data()
#
# plt.imshow(x_train[2],cmap='gray')
# plt.sho
# w()

img = Image.open('5.jpg')

# plt.imshow(img,cmap='gray')
# plt.show()

image = img.resize((28,28)).convert('L')

# plt.imshow(image,cmap='gray')
# plt.show()

image = np.array(image)

image = image.flatten()   #由于singleNet.h5模型是一维784输入，要对图片的形状改为相应的维度。
print(image.shape)
image = image.reshape(1,28,28,1)

model = load_model('myCNNMnist.h5')  #载入训练好的神经网络模型
print(model.predict_classes(image))  #使用训练好的神经模型进行图片的识别