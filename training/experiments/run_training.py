from training.common.metrics import SegmentationAccuracy
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.metrics import MeanIoU, CategoricalAccuracy
import collections

from training.data import SegmentationDataset
from training.model import BaseModel
from training.common import SegmentationLoss, SimpleLogCallback

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

weights_path= "/home/inmind/Documents/SidewalkDetection/training/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
num_epochs = 20
batch_size = 5
num_classes = 1


with tf.device("CPU"):
    
    train_dataset = SegmentationDataset(image_dir= train_image_dir, label_dir= train_seg_dir, batch_size= batch_size, resize= (224, 224))
    val_dataset = SegmentationDataset(image_dir= val_image_dir, label_dir= val_seg_dir, batch_size= -1, resize= (224, 224))
    
    model = BaseModel(num_classes, (None, 224, 224, 3))
    model.make_model()
    model.load_weights(weights_path)
    model.summary()

    metrics_to_print = collections.OrderedDict([("loss", "loss"), ("val_loss", "val_loss"),
                                            ("accuracy", "accuracy"), ("val_acc", "val_acc")])
   
    callback = [
        # Callback to stop training if val-loss stops improving
        #EarlyStopping(patience= 8, monitor= "val_loss", restore_best_weights= True),

        # Callback to save the model specifying the epoch and val-loss 
        #ModelCheckpoint(filepath= "training/logs/ckpts/weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5", period= 5),

        # Callback to log the graph, losses and metrics into tensorboard
        #TensorBoard(log_dir= "training/logs/tensorboard", histogram_freq= 0, write_graph= True),

        # Callback to log metrics at end of each epoch (if verbose=0)
        SimpleLogCallback(metrics_to_print, val= True,val_data= val_dataset, num_epochs=num_epochs)    
    ]
    
    
    model.compile(optimizer= 'adam', loss= SegmentationLoss(), metrics= ["accuracy"])

    model_history = model.fit(train_dataset, batch_size= 5, epochs= num_epochs, 
                            callbacks= callback, verbose= 0)
    