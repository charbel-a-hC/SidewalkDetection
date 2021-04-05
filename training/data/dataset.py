import tensorflow as tf 
import numpy as np
import math
import os

from tensorflow.keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize

class SegmentationDataset (Sequence):
    def __init__ (self, image_dir, label_dir, batch_size, augmentation= False, resize= None):
        # Load the images and labels path to memory
        self.images, self.gt_segmentations = _get_paths(image_dir, label_dir)

        if batch_size == -1:
            self.batch_size= len(self.images)
        else:
            self.batch_size = batch_size
        self.augmentation = augmentation
        self.resize = resize
  
    def __len__ (self):
        return int(math.ceil(len(self.images) / self.batch_size))

    def get_batch_image(self, idx):
        batch_images = self.images[idx * self.batch_size : (idx + 1) * self.batch_size] 

        if self.resize:
            batch_images = np.array([resize(imread(file_name), self.resize) for file_name in batch_images])
        else:
            batch_images = np.array([imread(file_name) for file_name in batch_images])
        
        return batch_images
    
    def get_batch_gt_segmentation(self, idx):
        batch_gt_segmentations = self.gt_segmentations[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.resize:
            batch_gt_segmentations = np.array([resize(imread(file_name, as_gray= True), self.resize) for file_name in batch_gt_segmentations]) 
        else:
            batch_gt_segmentations = np.array([imread(file_name, as_gray= True) for file_name in batch_gt_segmentations]) 
        
        return batch_gt_segmentations
    
    def __getitem__(self, index):
        batch_x = self.get_batch_image(index)
        batch_y = self.get_batch_gt_segmentation(index)
        return batch_x, batch_y



def _get_paths(images_directory, labels_directory):
    indexes = [x.split(".")[0].split("_")[1] for x in os.listdir(images_directory)]

    images_path = []
    labels_path = []

    for dirpath,_,filenames in os.walk(images_directory):
        for i in range(len(filenames)):
            filename = f"{filenames[i].split('_')[0]}_{indexes[i]}.jpg"
            images_path.append(os.path.abspath(os.path.join(dirpath, filename)))
    
    for dirpath,_,filenames in os.walk(labels_directory):
        for i in range(len(filenames)):
            filename = f"{filenames[i].split('_')[0]}_{indexes[i]}.jpg"
            labels_path.append(os.path.abspath(os.path.join(dirpath, filename)))

    return images_path, labels_path

### Not yet implemented
def _augmentation_fn(image, gt_image=None):
    """
    Apply random transformations to augment the training images.
    :param images:      Images
    :return:            Augmented Images
    """

    original_shape = tf.shape(image)[-3:-1]
    num_image_channels = tf.shape(image)[-1]

    # If we decide to randomly flip or resize/crop the image, the same should be applied to
    # the label one so they still match. Therefore, to simplify the procedure, we stack the
    # two images together along the channel axis, before these random operations:
    if gt_image is None:
        stacked_images = image
        num_stacked_channels = num_image_channels
    else:
        stacked_images = tf.concat([image, tf.cast(gt_image, dtype=image.dtype)], axis=-1)
        num_stacked_channels = tf.shape(stacked_images)[-1]

    # Randomly applied horizontal flip:
    stacked_images = tf.image.random_flip_left_right(stacked_images)

    # Random cropping:
    random_scale_factor = tf.random.uniform([], minval=.8, maxval=1., dtype=tf.float32)
    crop_shape = tf.cast(tf.cast(original_shape, tf.float32) * random_scale_factor, tf.int32)
    if len(stacked_images.shape) == 3:  # single image:
        crop_shape = tf.concat([crop_shape, [num_stacked_channels]], axis=0)
    else:  # batched images:
        batch_size = tf.shape(stacked_images)[0]
        crop_shape = tf.concat([[batch_size], crop_shape, [num_stacked_channels]], axis=0)
    stacked_images = tf.image.random_crop(stacked_images, crop_shape)

    # The remaining transformations should be applied either differently to the input and GT images
    # (nearest-neighbor resizing for the label image VS interpolated resizing for the image),
    # or only to the input image, not the GT one (color changes, etc.). Therefore, we split them back:
    image = stacked_images[..., :num_image_channels]

    # Resizing back to expected dimensions:
    image = tf.image.resize(image, original_shape)

    # Random B/S changes:
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.75)
    image = tf.clip_by_value(image, 0.0, 1.0)  # keeping pixel values in check

    if gt_image is not None:
        gt_image = tf.cast(stacked_images[..., num_image_channels:], dtype=gt_image.dtype)
        gt_image = tf.image.resize(gt_image, original_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image, gt_image
    else:
        return image