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