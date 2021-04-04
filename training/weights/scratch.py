import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.ops.gen_math_ops import Mod
from tensorflow.keras.datasets import mnist
import numpy as np

class layerA (Layer):
    def __init__ (self, name= "layer1", **kwargs):
        super(layerA, self).__init__(name=name, **kwargs)
    
    def build(self, input_shape):
        self.conv_1 = Conv2D(filters= 34, kernel_size= (3, 3), strides= 1, padding= "same", activation= "relu")
        self.conv_2 = Conv2D(filters= 34, kernel_size= (3, 3), strides= 1, padding= "same", activation= "relu")
        self.max_pool = MaxPool2D(pool_size= (2, 2))
    
    def call(self, inputs):
        x = inputs
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.max_pool(x)
        return x

class ModelA (Model):
    def __init__ (self,name= "fcn", **kwargs):
        super(ModelA, self).__init__(name= name, **kwargs)


    def build(self, input_shape):

        self.layerA = layerA()

        self.flatten_1 = Flatten()

        self.dense_1 = Dense(10, 'softmax')
        super(ModelA, self).build(input_shape)

    def call (self, inputs):
        x = inputs
        x = self.layerA(x)
        x = self.flatten_1(x)
        x = self.dense_1(x)
        return x

class ModelB(Model):
    def __init__ (self, name='unet', **kwargs):
        super(ModelB, self).__init__(name= name, **kwargs)



"""
#model = model_main(model_name='fcn', input_shape= (None, 28, 28, 1))
model.make_model()
model.summary()
model.load_weights("")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

with tf.device("CPU"):
    x_train = x_train.reshape((x_train.shape[0],28, 28, 1))
    x_train_processed = x_train[:200, :, :]/255.0
    y_train_processed = y_train[:200]
    conv1_weights = model().layers[0].get_weights()
    
    conv1_W = conv1_weights[0]
    conv1_b = conv1_weights[1]

    print(conv1_W.shape)
    print(conv1_b.shape)
    print(conv1_W)
    print("***"*10)
    print(conv1_b)

    model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ["accuracy"])
    model_history = model.fit(x_train_processed, y_train_processed, epochs= 10)
"""

"""
with tf.device("CPU"):
    #input_tensor = tf.random.uniform(shape= [2, 28, 28, 1], minval= 0, maxval= 1)
    #convblock = Conv2Block(12, 1, (1,1), 'same', "hello")
    #output = convblock(input_tensor)
    #weights= tf.random.uniform(shape=[2, 1, 1, 1, 12])
    #print(convblock.layers[0].output)
    #print(convblock.layers[0].set_weights(weights))
    #print(output)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((x_train.shape[0],28, 28, 1))
    x_train_processed = x_train[:200, :, :]/255.0
    y_train_processed = y_train[:200]


    model = FCN()
    model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ["accuracy"])
    model.fit(x_train_processed, y_train_processed, epochs= 100)

"""
