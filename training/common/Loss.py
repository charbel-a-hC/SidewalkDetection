import tensorflow as tf
from training.common.common import prepare_data_for_segmentation_loss

class SegmentationLoss(tf.losses.SparseCategoricalCrossentropy):
    def __init__(self, ignore_value=CITYSCAPES_IGNORE_VALUE, 
                 from_logits=False, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'):
        super().__init__(from_logits=from_logits, reduction=reduction, name=name)
        self.ignore_value = ignore_value
    
    def _prepare_data(self, y_true, y_pred):
        num_classes = y_pred.shape[-1]
        y_true, y_pred = prepare_data_for_segmentation_loss(y_true, y_pred,
                                                            num_classes=num_classes, 
                                                            ignore_value=self.ignore_value)
        return y_true, y_pred
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = self._prepare_data(y_true, y_pred)
        loss = super().__call__(y_true, y_pred, sample_weight)
        return loss