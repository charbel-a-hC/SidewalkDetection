import tensorflow as tf
from tensorflow.keras.losses import Reduction, BinaryCrossentropy

from training.common.common import prepare_data_for_segmentation_loss

class SegmentationLoss(BinaryCrossentropy):
    def __init__(self,
                from_logits= False, reduction=Reduction.SUM_OVER_BATCH_SIZE, name='loss'):
        super().__init__(from_logits= from_logits, reduction=reduction, name=name)
    
    def _prepare_data(self, y_true, y_pred):
        num_classes = y_pred.shape[-1]
        y_true, y_pred = prepare_data_for_segmentation_loss(y_true, y_pred,
                                                            num_classes=num_classes)
        return y_true, y_pred
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = self._prepare_data(y_true, y_pred)
        loss = super().__call__(y_true, y_pred, sample_weight)
        return loss