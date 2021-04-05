from hashlib import new
from training.model.fcn import FCN
from training.model.unet import Unet
from training.weights.load_weights import load_vgg_weights

import numpy as np 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import cv2


class BaseModel:
    def __init__ (self, num_classes: int, input_shape= None, model_name= 'fcn'):
        self.model_name = model_name
        self.model = None
        self._make_model = False
        
        self.input_shape = input_shape
        self.num_classes = num_classes

    def make_model(self):
        if self.model_name == 'fcn':
            self.model = FCN(num_output_channels= self.num_classes)
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
    def compile(self,
                optimizer= "rmsprop",
                loss= None, 
                metrics= None,
                loss_weights= None,
                sample_weight_mode= None,
                weighted_metrics= None,
                **kwargs):
                
        self.model.compile(
            optimizer= optimizer,
            loss= loss,
            mterics= metrics,
            loss_weights= loss_weights,
            sample_weight_mode= sample_weight_mode,
            weighted_metrics= weighted_metrics,
            **kwargs
        )

    ### Training
    def fit(self,
        x,
        y,
        batch_size,
        epochs,
        verbose=1,
        callbacks=None,
        validation_data=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_freq=1,):

        self.model.fit(
            x,
            y,
            batch_size=batch_size,
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
    def predict(self, image):
        return self.model.predict(image)

    ### Save weights
    def save_weights(self, filepath, overwrite= None, save_format= None):
        self.model.save_weights(
            filepath= filepath,
            overwrite= overwrite,
            save_format= save_format)

    def __call__(self):
        return self.model

