#! /usr/bin/env python

import sys
import os
import keras.metrics
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import tensorflow as tf
import pickle
import time
base_dir = os.path.abspath('../../')
sys.path.insert(0, '%s/utils'%base_dir)
sys.path.insert(0, '%s/common'%base_dir)
from covrew_data_generator import data_generator
from viz_net_model import viz_net_model
#import custom_loss_functions
from custom_loss_functions import switch_loss_fn
from custom_metrics import rrmse

import yaml
#from data_inspector import data_inspector

def main(argv):

    if(len(sys.argv) != 2):
        print("num sys argv %d"%len(sys.argv))
        print("usage: %s <config file path>" % sys.argv[0])
        return -1

    #with open('/home/hankm/results/neuro_exploration_res/viz_net/params/viznet_params_cont.yaml') as f:
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_handler = data_generator(name='data_handler', **config)

    #load conf file
    tf.random.set_seed(42)
    out_types = ( tf.float32, tf.float32 )
    out_shapes= ( (data_handler.gmheight_p, data_handler.gmwidth_p, data_handler.input_channels), (data_handler.gmheight_p, data_handler.gmwidth_p, data_handler.output_channels) )
    val_out_shapes = ( (None, data_handler.gmheight_p, data_handler.gmwidth_p, data_handler.input_channels),
                       (None, data_handler.gmheight_p, data_handler.gmwidth_p, data_handler.output_channels) )
    start = time.time()
    worlds_idx = [0, 1, 2, 3, 4] # 5 is saved for validation
    train_ds = tf.data.Dataset.from_generator(lambda: data_handler.train_data_generator(world_idxs=worlds_idx, num_rounds= data_handler.num_rounds,
                                            shuffle=True, data_augment=True), output_types=out_types, output_shapes=out_shapes).batch(data_handler.batch_size)
    np_val_round_idxs = np.asarray(range(0, 49))
    val_ds = tf.data.Dataset.from_generator(lambda: data_handler.val_data_generator(widx=5, np_round_idxs=np_val_round_idxs), output_types=out_types,
                                              output_shapes= val_out_shapes )
    # model arch
    model = viz_net_model(**config)
    #model.summary()
    #train specs
    outmodelfile = config['training_configs']['outmodelfile']
    outhistfile  = config['training_configs']['outhistfile']
    loss_fn_txt  = config['training_configs']['loss_fn']
    loss_fn = switch_loss_fn(loss_fn_txt)

    check_point_callback = tf.keras.callbacks.ModelCheckpoint(filepath=outmodelfile, verbose=1, save_best_only=True)
    my_callbacks = [check_point_callback]
    lr  = config['training_configs']['lr']
    if (lr > 0):
        opt = keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = keras.optimizers.Adam()
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=0.0001)
        my_callbacks.append(reduce_lr)

    model.compile(optimizer=opt, loss=loss_fn, metrics=["mae"])    #binary_crossentropy
    print("********************************************* \n callback fns: \n", *my_callbacks, sep=", ")
    print("\n ********************************************* \n")
    # start training
    num_epochs = int(config['training_configs']['num_epochs'])
    hist = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=my_callbacks, verbose=1)
    model.save(outmodelfile)
    with open(outhistfile, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)


if __name__ == '__main__':
   main(sys.argv)
