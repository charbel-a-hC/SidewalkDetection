import tensorflow as tf

def get_mask_for_valid_labels(y_true, num_classes, ignore_value=255):
    """
    Build a mask for the valid pixels, i.e. those not belonging to the ignored classes.
    :param y_true:        Ground-truth label map(s) (each value represents a class trainID)
    :param num_classes:   Total nomber of classes
    :param ignore_value:  trainID value of ignored classes (`None` if ignored none)
    :return:              Binary mask of same shape as `y_true`
    """
    mask_for_class_elements = y_true < num_classes
    mask_for_not_ignored = y_true != ignore_value
    mask = mask_for_class_elements & mask_for_not_ignored
    return mask


def prepare_data_for_segmentation_loss(y_true, y_pred, num_classes=10, ignore_value=255):
    """
    Prepare predicted logits and ground-truth maps for the loss, removing pixels from ignored classes.
    :param y_true:        Ground-truth label map(s) (e.g., of shape B x H x W)
    :param y_pred:        Predicted logit map(s) () (e.g., of shape B x H x W x N, N number of classes)
    :param num_classes:   Number of classes
    :param ignore_value:  trainID value of ignored classes (`None` if ignored none) 
    :return:              Tensors edited, ready for the loss computation
    """

    with tf.name_scope('prepare_data_for_loss'):
        # Flattening the tensors to simplify the following operations:
        if len(y_pred.shape) > (len(y_true.shape) - 1):
            y_pred = tf.reshape(y_pred, [-1, num_classes])
        else:
            y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])

        if ignore_value is not None:
            # To compare only on the considered class, we remove all the elements in the images
            # belonging to the ignored ones.
            # For that, we first compute the mask of the pixels belonging to valid labels:
            mask_for_valid_labels = get_mask_for_valid_labels(
                y_true, num_classes, ignore_value=ignore_value)
    
            # Then we use this mask to remove all pixels/elements not belonging to valid classes:
            y_true = tf.boolean_mask(y_true, mask_for_valid_labels, axis=0, name='gt_valid')
            y_pred = tf.boolean_mask(y_pred, mask_for_valid_labels, axis=0, name='pred_valid')

    return y_true, y_pred