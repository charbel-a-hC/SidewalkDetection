import tensorflow as tf 
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D
 
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