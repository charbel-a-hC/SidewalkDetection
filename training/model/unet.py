import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Conv2DTranspose, UpSampling2D, ReLU, Dense, AveragePooling2D
from tensorflow.keras.datasets import mnist

class ConvLayer(Layer) :
    def __init__(self, nf, ks=3, s=2, **kwargs):
        self.nf = nf
        self.grelu = ReLU()
        self.conv = (Conv2D(filters     = nf,
                            kernel_size = ks,
                            strides     = s,
                            padding     = "same",
                            use_bias    = False,
                            activation  = "linear"))
        super(ConvLayer, self).__init__(**kwargs)

    def rsub(self): return -self.grelu.sub
    def set_sub(self, v): self.grelu.sub = -v
    def conv_weights(self): return self.conv.weight[0]

    def build(self, input_shape):
        # No weight to train.
        super(ConvLayer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0],
                        input_shape[1]/2,
                        input_shape[2]/2,
                        self.nf)
        return output_shape

    def call(self, x):
        return self.grelu(self.conv(x))

    def __repr__(self):
        return f'ConvLayer(nf={self.nf}, activation={self.grelu})'

class ConvModel(tf.keras.Model):
    def __init__(self, nfs, input_shape, output_shape, use_bn=False, use_dp=False):
        super(ConvModel, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp


        # backbone layers
        self.convs = [ConvLayer(nfs[0], s=1, input_shape=input_shape)]
        self.convs += [ConvLayer(nf) for nf in nfs[1:]]
        # classification layers
        self.convs.append(AveragePooling2D())
        self.convs.append(Dense(output_shape, activation='softmax'))

    def call(self, inputs):
        for layer in self.convs: inputs = layer(inputs)
        return inputs

model = ConvModel([128, 256], (1, 28, 28, 1), 10)
model.build(input_shape= (1, 28, 28, 1))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()