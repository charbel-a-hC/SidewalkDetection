import random
import collections
import tensorflow as tf
from training.data import SegmentationDataset
from training.common import SegmentationLoss
from training.common.metrics import SegmentationAccuracy

class SegmentationLog(tf.keras.callbacks.Callback):
    """ Keras callback for simple, denser console logs."""

    def __init__(self, metrics_dict, txt_log_path, val= False, val_data: SegmentationDataset = None, num_epochs='?', log_frequency=1,
                 metric_string_template='\033[1m[[name]]\033[0m = \033[94m{[[value]]:5.3f}\033[0m'):
        """
        Initialize the Callback.
        :param metrics_dict:            Dictionary containing mappings for metrics names/keys
                                        e.g. {"accuracy": "acc", "val. accuracy": "val_acc"}
        :param num_epochs:              Number of training epochs
        :param log_frequency:           Log frequency (in epochs)
        :param metric_string_template:  (opt.) String template to print each metric
        """
        super().__init__()

        self.metrics_dict = collections.OrderedDict(metrics_dict)
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency

        # We build a format string to later print the metrics, (e.g. "Epoch 0/9: loss = 1.00; val-loss = 2.00")
        log_string_template = 'Epoch {0:2}/{1}: '
        separator = '; '

        i = 2
        for metric_name in self.metrics_dict:
            templ = metric_string_template.replace('[[name]]', metric_name).replace('[[value]]', str(i))
            log_string_template += templ + separator
            i += 1

        # We remove the "; " after the last element:
        log_string_template = log_string_template[:-len(separator)]
        self.log_string_template = log_string_template

        self.val_data = val_data
        self.val = val
        self.txt_log_path = txt_log_path

    def on_train_begin(self, logs=None):
        print("Training: \033[92mstart\033[0m.")
        

    def on_train_end(self, logs=None):
        print("Training: \033[91mend\033[0m.")

    def on_epoch_end(self, epoch, logs={}):
        if self.val:
            random_batch = random.randint(0, len(self.val_data)-1)
            y_pred = self.model.predict(self.val_data[random_batch][0])
            y_true = self.val_data[random_batch][1]
            total_loss = SegmentationLoss()(y_true= y_true, y_pred= y_pred)
            accuracy = SegmentationAccuracy()(y_true= y_true, y_pred= y_pred)

            logs["val_loss"] = total_loss.numpy()
            logs["val_acc"] = accuracy.numpy()

        output = ""

        if (epoch - 1) % self.log_frequency == 0 or epoch == self.num_epochs:
            values = [logs[self.metrics_dict[metric_name]] for metric_name in self.metrics_dict]
            output = self.log_string_template.format(epoch+1, self.num_epochs, *values)
            print(output)

        with open(f"{self.txt_log_path}/logs.txt", "a") as f:
            f.writelines(f"Epoch {epoch+1}/{self.num_epochs}: loss= {self.metrics_dict['loss']}; val_loss= {logs['val_loss']}; accuracy= {self.metrics_dict['accuracy']}; val_acc= {logs['val_acc']}")
