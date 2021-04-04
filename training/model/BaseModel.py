from training.model.fcn import FCN
from training.model.unet import Unet
from training.weights.load_weights import load_vgg_weights

import numpy as np 
import tensorflow as tf


class BaseModel:
    def __init__ (self, model_name= 'fcn', input_shape= None):
        self.model_name = model_name
        self.model = None
        self._make_model = False
        self.input_shape = input_shape
    
    def make_model(self):
        if self.model_name == 'fcn':
            self.model = FCN()
            self.model.build(input_shape= self.input_shape)
            self._make_model = True
        elif self.model_name == "unet":
            self.model = Unet()
            self.model.build(input_shape= self.input_shape)

    
    def summary(self):
        self.model.summary()
    
    def load_weights(self, weights_path):
        if self.model_name == 'fcn':
            weights = load_vgg_weights(weights_path)
            for layer in self.model.layers[:5]:
                layer.set_weights(weights[layer.name])
                
    ### Compile
    def compile(self, optimizer, loss, metrics, callbacks=None):
        self.model.compile(
            optimizer= optimizer,
            loss= loss,
            metrics= metrics
        )

    ### Training
    def fit(self,
        data_set,
        epochs,
        verbose=1,
        callbacks=None,
        validation_data=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_freq=1,):

        self.model.fit(
            data_set,
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=0.0,
            validation_data=validation_data,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            #validation_batch_size=None,
            validation_freq=validation_freq,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
    )

    ### Predict
    

    ### Save weights


    def __call__(self):
        return self.model

with tf.device("CPU"):
    model = BaseModel("fcn", (None, 96, 32, 3))
    model.make_model()
    model.summary()
    file_ = "training/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model.load_weights(file_)
    

