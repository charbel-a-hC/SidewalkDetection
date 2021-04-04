import h5py
import numpy as np
import os 

os.environ["H5PY_DEFAULT_READONLY"] = "1"

file_ = "training/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# VGG16
def load_vgg_weights (filepath):
    layers = {}
    f = h5py.File(filepath)
    layer_names = list(f.keys())
    for i in range(1, 6):
        blocks = [x for x in layer_names if x.split('_')[0] == f"block{i}"]
        blocks = list(filter(lambda x: 'pool' not in x, blocks ))

        temp_values = []
        for block in blocks:
            conv_w = f[block][f"{block}_W_1:0"].value
            conv_b = f[block][f"{block}_b_1:0"].value
            temp_values.append(conv_w)
            temp_values.append(conv_b)
        
        layers[f"conv_{len(blocks)}_block_{i}"] = temp_values
    f.close()
    return layers
