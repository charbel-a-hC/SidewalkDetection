import tensorflow as tf
from training.common.common import prepare_data_for_segmentation_loss

class SegmentationAccuracy(tf.metrics.Accuracy):
    def __init__(self, name='acc', dtype=None):
        super().__init__(name=name, dtype=dtype)


    def __call__(self, y_true, y_pred, sample_weight=None):
        num_classes = y_pred.shape[-1]
        y_true, y_pred = prepare_data_for_segmentation_loss(y_true, y_pred,
                                                            num_classes=num_classes)
        # And since tf.metrics.Accuracy needs the label maps, not the one-hot versions,
        # we adapt accordingly:
        y_pred = tf.argmax(y_pred, axis=-1)
        
        return super().__call__(y_true, y_pred, sample_weight)