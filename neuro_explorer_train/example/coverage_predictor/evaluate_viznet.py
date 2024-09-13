
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import tensorflow as tf
import time
base_dir = os.path.abspath('../../')
sys.path.insert(0, '%s/utils'%base_dir)
sys.path.insert(0, '%s/common'%base_dir)
from covrew_data_generator import data_generator
from custom_metrics import switch_metics
import yaml

def main(argv):
    #with open('/home/hankm/results/neuro_exploration_res/viz_net/params/viznet_params_cont.yaml') as f:
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_handler = data_generator(name='data_handler', **config)
    #train specs
    modelfile = config['training_configs']['outmodelfile']
    histfile  = config['training_configs']['outhistfile']
    metric_txt = config['testing_configs']['metric_fn']
    metric_fn = switch_metics(metric_txt)
    # load model
    tf.random.set_seed(42)
    out_types = ( tf.float32, tf.float32 )
    out_shapes = ( (None, data_handler.gmheight_p, data_handler.gmwidth_p, data_handler.input_channels),
                   (None, data_handler.gmheight_p, data_handler.gmwidth_p, data_handler.output_channels) )
    start = time.time()
    np_test_round_idx = np.asarray(range(0, 100)) #np.asarray(range(50, 100))
    # start = time.time()
    val_ds   = tf.data.Dataset.from_generator(lambda: data_handler.test_data_generator(widx=5, np_round_idxs=np_test_round_idx), output_types=out_types, output_shapes=out_shapes)
    model = tf.keras.models.load_model(modelfile, compile=False)  # custom_objects={ 'weighted_cce': weighted_cce(CCE_WEIGHTS) })
    #model.compile(optimizer='adam', loss=loss_fn, metrics=CustomRRMSE(output_channels=data_handler.output_channels) )  # binary_crossentropy
    model.compile(metrics=['mae', metric_fn])
    model.evaluate(val_ds)

if __name__ == '__main__':
   main(sys.argv)