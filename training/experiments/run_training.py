import tensorflow as tf
from training.model import BaseModel

# Instantiate a BaseModel mdoel -> either an FCN-8s or Unet model
model = BaseModel(1, (None, 1920, 1024, 3))
# Make and build the model
# In order to build the model, the input dims must be multiples of 32
model.make_model()
# Show model architechture summary
model.summary()
# Load pre-trained weights for VGG16 feature extractor
model.load_weights("training/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
# Compile the model
model.compile(optimizer= "adam", loss= , metrics= )