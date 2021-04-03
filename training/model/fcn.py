# Build a VGG16 feature extractor and load the corresponding weights without the top layers for classification

### Conv Block
### VGG-16 feature extractor
### Decoder FCN-8s


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Conv2DTranspose, UpSampling2D, ReLU
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.layers.core import Dense, Flatten

# ConvBlock -- conv(N, k, s, p) - conv(N, k, s, p) - maxpool(2, 2)
class Conv2Block (Layer): 
    def __init__ (self, filters: int, kernel_size, stride, padding, name: str, **kwargs):
        super(Conv2Block, self).__init__(name= name, **kwargs)

        self.filters = filters
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif all(map(lambda x: isinstance(x, int), kernel_size)):
            self.kernel_size = kernel_size
        else:
            raise Exception("Wrong kernel_size type: (int, int) or int")
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif all(map(lambda x: isinstance(x, int), stride)):
            self.stride = stride
        else:
            raise Exception("Wrong stride type: (int, int) or int")
        
        self.padding = padding

    def build(self, input_shape): 

        self.conv_1 = Conv2D(filters= self.filters, kernel_size= self.kernel_size, strides= self.stride, padding= self.padding, name="conv_1", input_shape= input_shape)
        self.conv_2 = Conv2D(filters= self.filters, kernel_size= self.kernel_size, strides= self.stride, padding= self.padding, name= "conv_2")
        self.max_pool = MaxPool2D(pool_size= (2, 2), strides= (2, 2))
        super(Conv2Block, self).build(input_shape)

    def call(self, input_tensor):
        x = input_tensor
        
        x = self.conv_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
    
        return x

# ConvBlock -- conv(N, k, s, p) - conv(N, k, s, p) - conv(N, k, s, p) - maxpool(2, 2)
class Conv3Block (Layer): 
    def __init__ (self, filters: int, kernel_size, stride, padding, name: str):
        super(Conv3Block, self).__init__(name= name)

        self.filters = filters
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif all(map(lambda x: isinstance(x, int), kernel_size)):
            self.kernel_size = kernel_size
        else:
            raise Exception("Wrong kernel_size type: (int, int) or int")
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif all(map(lambda x: isinstance(x, int), stride)):
            self.stride = stride
        else:
            raise Exception("Wrong stride type: (int, int) or int")
        
        self.padding = padding

    def build(self, input_shape): 

        self.conv_1 = Conv2D(filters= self.filters, kernel_size= self.kernel_size, strides= self.stride, padding= self.padding, name="conv_1")
        self.conv_2 = Conv2D(filters= self.filters, kernel_size= self.kernel_size, strides= self.stride, padding= self.padding, name= "conv_2")
        self.conv_3 = Conv2D(filters= self.filters, kernel_size= self.kernel_size, strides= self.stride, padding= self.padding, name= "conv_3")
        self.max_pool = MaxPool2D(pool_size= (2, 2), strides= (2, 2))
        super(Conv3Block, self).build(input_shape)

    def call(self, input_tensor):
        x = input_tensor
        
        x = self.conv_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = tf.nn.relu(x)
        x = self.conv_3(x)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
    
        return x
    
class FCN (Model):
    def __init__(self):
        super(FCN, self).__init__()

    def build (self, input_shape):

        ### Encoder
        # Set pre-trained weights for VGG16 feature extractor part of network
        self.conv_2_block_1 = Conv2Block(64, 3, 1, "same", "conv_2_block_1", input_shape= input_shape)
        
        #self.conv_block_1.layers[0].set_weights()
        #self.conv_2_block_2 = Conv2Block(64, 3, 1, "same", "conv_2_block_2")
        # set weights
        #self.conv_3_block_1 = Conv3Block(128, 3, 1, "same", "conv_3_block_1")

        #self.conv_3_block_2 = Conv3Block(256, 3, 1, "same", "conv_3_block_2")

        #self.conv_3_block_3 = Conv3Block(512, 3, 1, "same", "conv_3_block_2")

        #self.conv_3_block_4 = Conv3Block(512, 3, 1, "same", "conv_3_block_3")
        
        ### Decoder
        #self.conv_

        self.flatten_1 = Flatten()
        self.dense_1 = Dense(10, activation= "softmax")
        
        super(FCN, self).build(input_shape)

    def call (self, input_tensor):
        x = input_tensor
        x = self.conv_2_block_1(x)
        x = self.flatten_1(x)
        x = self.dense_1(x)
        return x

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

    model.build(input_shape= (None, 28, 28, 1))
    model.compile(optimizer= 'sgd', loss= tf.keras.losses.SparseCategoricalCrossentropy(), metrics= ['accuracy'])
    model.summary()
    #model.fit(x_train_processed, y_train_processed, epochs= 10)

    #print(model(x_train_processed[:2, :, :]))
    #print(y_train_processed[:2])