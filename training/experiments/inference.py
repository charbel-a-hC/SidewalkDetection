#### Inference Script
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import cv2

def vis_images (image_raw: np.ndarray, image_mask: np.ndarray, mask_alpha: float): 
    
    fig = plt.figure(figsize= (10, 10))
    
    # Raw image
    fig.add_subplot(2, 3, 1)
    plt.imshow(image_raw)
    plt.title('Raw image')
    
    # Segmentation mask
    fig.add_subplot(2, 3, 3)
    plt.imshow(image_mask)
    plt.title('Segmentation Mask')
    
    # Raw + mask
    fig.add_subplot(2, 3, 5)
    plt.imshow(cv2.addWeighted(image_raw, 1, image_mask, mask_alpha, 0))
    plt.title('Raw + mask')
    plt.show()

# This script performs inference on single images
# Should be replaced with feed from camera

input_image = imread("dataset/test/image/Raw_74.jpg")
gt_segmentation = imread("dataset/train/label/Label_121.jpg")

# Expected input to be (1, 224, 224, 3)
input_image_processed = cv2.resize(input_image, (224, 224))
input_image_processed = np.reshape(input_image_processed, (1, *input_image_processed.shape))/255.0

# Load a new model from SaveModel
new_model = tf.keras.models.load_model("training/experiments/logs/weights/sidewalk-detect", compile= False)

# Expected output to be (1, 224, 224, 1)
output = new_model.predict(input_image_processed)
output = np.reshape(output, (224, 224))
# Resize to size of input image from camera
dsize = input_image.shape[:2]
output = cv2.resize(output, dsize[::-1])

# Show the output, the input and the weighted sum
output = cv2.normalize(output, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

output = output.astype(np.uint8)
output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

# Postprocessing for the output (sharpen the image)
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])# applying the sharpening kernel to the input image & displaying it.
sharpened_output = cv2.filter2D(output, -1, kernel_sharpening)

sharpened_output[sharpened_output <= 125] = 0
sharpened_output[sharpened_output > 125] = 255

## Set output color: R - G - B
sharpened_output[:, :, 0] = 0
sharpened_output[:, :, 1] = 0
#sharpened_output[:, :, 1] = 0
plt.imshow(sharpened_output)

## Prediction
vis_images(image_raw= input_image, image_mask= sharpened_output, mask_alpha= 0.5)

## Ground Truth
gt_segmentation[:, :, 0] = 0
gt_segmentation[:, :, 1] = 0
vis_images(image_raw= input_image, image_mask= gt_segmentation, mask_alpha= 0.5)