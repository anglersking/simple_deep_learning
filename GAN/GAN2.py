import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

(train_images,train_lables),(_,_)  = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0],28,28,1)

train_images = train_images.astype('float32')  #转为浮点数

train_images = (train_images -127.5)/127.5  #归一化为（-1，1）范围的数据


BATCH_SIZE = 256
BUFFER_SIZE = 60000

datasets = tf.data.Dataset.from_tensor_slices(train_images)
datasets = datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)  #做shuffle


#做一个生成器
def generator_mode():

    model = keras.Sequential()
    model.add(layers.Dense(256,input_shape=(100,),use_bias=False)) #输入噪声
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(512,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(28*28*1,use_bias=False,activation='tanh'))
    model.add(layers.BatchNormalization())

    model.add(layers.Reshape((28,28,1))) #生成一个28*28*1的图片

    return  model

#判别器
def discriminator_model():

    model = keras.Sequential()
    model.add(layers.Flatten())

    model.add(layers.Dense(512,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(256,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(1)) #输出一维的结果 真1假0

    return model

#定义2分类的交叉熵代价函数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) #二元交叉熵函数

#定义判别的loss 计算真实图片的loss和假图片的loss
def discreminator_loss(real_image,fake_image):
    #希望真的图片都判别为1，这样real_loss为0
    real_loss = cross_entropy(tf.ones_like(real_image),real_image)
    #希望假的图片都判别为0，这样fake_loss为0
    fake_loss = cross_entropy(tf.zeros_like(fake_image),fake_image)
    #返回真的loss和假的loss
    return real_loss+fake_loss


#定义生成器的loss
def generator_loss(fake_out):
    #希望生成器的假图片都判别为真
    loss = cross_entropy(tf.ones_like(fake_out),fake_out)
    return loss

#定义判别器和生成器的优化器
generator_opt = tf.keras.optimizers.Adam(3e-4)
discriminator_opt = tf.keras.optimizers.Adam(1e-4)


#创建我们的生成器模型
generator = generator_mode()
#创建一个判别器的模型
discriminator = discriminator_model()


noise_dim = 100

#定义训练模型
def train_step(images): #传入真实的图片

    noise = tf.random.normal([BATCH_SIZE,noise_dim])

    #使用tensorflow自带的梯度下降函数进行迭代运算
    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
        #使用噪声输出产生假的图片
        gen_image = generator(noise,training=True)

        #使用真实图片传入判别器
        real_out = discriminator(images,training=True)
        #传入假的图片到判别器
        fake_out = discriminator(gen_image,training=True)

        #计算生成器的loss
        gen_loss = generator_loss(fake_out)

        #计算判别器的loss
        disc_loss = discreminator_loss(real_out,fake_out)
    #传入生成器的loss，计算生成器的权值进行调整
    gradient_gen = gen_tape.gradient(gen_loss,generator.trainable_variables)
    #传入判别器的loss，计算判别器的权重进行调整
    gradient_disc = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

    #对生成器权值进行调整
    generator_opt.apply_gradients(zip(gradient_gen,generator.trainable_variables))
    #对判别器权值进行调整
    discriminator_opt.apply_gradients(zip(gradient_disc,discriminator.trainable_variables))

#产生一个随机数噪声，用于训练过程生成图片
seed = tf.random.normal([16,noise_dim])

#生成图片函数，第一个参数是生成器的模型，第二个参数是传入的噪声
def genrator_plt_image(gen_mode,test_noise):
    pred_image = gen_mode(test_noise,training=False)
    fig = plt.figure(figsize=(4,4)) #创建一个4*4的图片

    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(pred_image[i,:,:,0],cmap='gray')
        plt.axis('off')
    plt.show()

#创建训练函数，第一个参数是tf的dataset，第二个参数训练次数
def train(dataset,epochs):
    for epoch in range(epochs):
        for image_batch in datasets:
            train_step(image_batch)
            print('.',end='')

        genrator_plt_image(generator,seed)

#进行模型训练
train(datasets,100)

