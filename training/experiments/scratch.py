import tensorflow as tf
from training.data import SegmentationDataset
from training.model import BaseModel
from training.common import SegmentationLoss

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_image = "/home/inmind/Documents/SidewalkDetection/dataset/train/image"
train_seg = "/home/inmind/Documents/SidewalkDetection/dataset/train/label"
weights_path= "/home/inmind/Documents/SidewalkDetection/training/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"



## Test segmentationdatast outputs
## Test output of augmentations
## implement augmentation
## implement shuffling of data

with tf.device("CPU"):
    """
    # Model trained to 86% accuracy on the training set but with final loss= 1586582
    a = SegmentationDataset(image_dir= train_image, label_dir= train_seg, batch_size= 5, resize= (224, 224))
    model = BaseModel(1, (None, 224, 224, 3))
    model.make_model()
    model.load_weights(weights_path)
    model.summary()

    model.compile(optimizer= 'adam', loss= SegmentationLoss(), metrics= ["accuracy"])
    model.fit(a, batch_size= 5, epochs= 2)
    """