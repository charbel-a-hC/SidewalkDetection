import tensorflow as tf 
from tensorflow.keras.layers import Layer, Conv2D, Dropout, Conv2DTranspose

class FCNBlock (Layer):
    def __init__ (self, name= "FCNBlock", **kwargs):
        super(FCNBlock, self).__init__(name= name, **kwargs)
    
    def build (self, input_shape):
        self.conv_1 = Conv2D(filters= 4086, kernel_size= 7, padding= "same", activation= "relu")
        self.dropout_1 = Dropout(0.5)
        self.conv_2 = Conv2D(filters= 4086, kernel_size= 1, padding="same",
                                activation= "relu")
        self.dropout_2 = Dropout(0.5)
        self.conv_3 = Conv2D(filters=3, kernel_size=1, padding='same',
                      activation=None)
        self.conv_transpose_1 = Conv2DTranspose(filters= 3, kernel_size= 4, strides= 2,
                                                use_bias= False, padding= "same", activation= "relu")
        
        super(FCNBlock, self).build(input_shape)
    
    def call (self, inputs):
        x = inputs
        x = self.conv_1(x)
        x = self.dropout_1(x)
        x = self.conv_2(x)
        x = self.dropout_2(x)
        x = self.conv_3(x)
        x = self.conv_transpose_1(x)
        return x