import os
from training.common.metrics import SegmentationAccuracy
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.metrics import MeanIoU, CategoricalAccuracy
import collections

from training.data import SegmentationDataset
from training.model import BaseModel
from training.common import SegmentationLoss, SegmentationLog, prepare_data_for_segmentation_loss

## Setup validation callback for tensorboard
## Setup parser for running command line from CLI
## Test segmentationdatast outputs
## Test output of augmentations
## implement augmentation
## implement shuffling of data
## implement callback for tensorboard visualization

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_image_dir = "/home/inmind/Documents/SidewalkDetection/dataset/train/image"
train_seg_dir = "/home/inmind/Documents/SidewalkDetection/dataset/train/label"
val_image_dir = "/home/inmind/Documents/SidewalkDetection/dataset/val/image"
val_seg_dir = "/home/inmind/Documents/SidewalkDetection/dataset/val/label"

print(len(os.listdir(val_image_dir)))
print(100*"***")
print(100*"***")
print(100*"***")
print(45//10)

weights_path= "/home/inmind/Documents/SidewalkDetection/training/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
num_epochs = 20
batch_size = 10
num_classes = 1


with tf.device("CPU"):
    
    train_dataset = SegmentationDataset(image_dir= train_image_dir, label_dir= train_seg_dir, batch_size= batch_size, resize= (224, 224))
    val_dataset = SegmentationDataset(image_dir= val_image_dir, label_dir= val_seg_dir, batch_size= batch_size, resize= (224, 224))
    print(len(val_dataset))
    model = BaseModel(1, (None, 224, 224, 3))
    model.make_model()
    image = val_dataset[4][0]
    print(image.shape)

    """
    y_true = train_dataset[1][1][:10]
    y_pred = model(image)

    y_true, y_pred = prepare_data_for_segmentation_loss(y_true, y_pred)
    total_loss = SegmentationLoss()(y_true, y_pred).numpy()
    accuracy = SegmentationAccuracy()(y_true, y_pred).numpy()

    print(y_pred)
    print(y_true)
    print(total_loss)
    print(accuracy)
    """
