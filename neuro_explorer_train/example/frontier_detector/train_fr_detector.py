#! /usr/bin/env python

import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import tensorflow as tf
import pickle
import time
base_dir = os.path.abspath('../../')
sys.path.insert(0, '%s/utils'%base_dir)
sys.path.insert(0, '%s/common'%base_dir)
from fr_data_generator import fr_data_generator
from unet_model import unet_model
import yaml

def main(argv):

    if(len(sys.argv) != 2):
        print("num sys argv %d"%len(sys.argv))
        print("usage: %s <config file path>" % sys.argv[0])
        return -1

    #with open('%s/configs/frnet_params.yaml'%base_dir) as f:
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_handler = fr_data_generator(name='data_handler', **config)

    #load conf file
    tf.random.set_seed(42)
    out_types = ( tf.float32, tf.float32 )
    out_shapes= ( (data_handler.gmheight_p, data_handler.gmwidth_p, data_handler.input_channels), (data_handler.gmheight_p, data_handler.gmwidth_p, data_handler.output_channels) )
    val_out_shape = ( (1,data_handler.gmheight_p, data_handler.gmwidth_p, data_handler.input_channels), (1,data_handler.gmheight_p, data_handler.gmwidth_p, data_handler.output_channels) )
    start = time.time()
    worlds_idx = [0, 1, 2, 3, 4] # 5 is saved for validation
    train_ds = tf.data.Dataset.from_generator(lambda: data_handler.train_data_generator(world_idxs=worlds_idx, num_rounds=data_handler.num_rounds,
                                            shuffle=True, data_augment=True), output_types=out_types, output_shapes=out_shapes).batch(data_handler.batch_size)
    np_val_round_idxs = np.asarray(range(0, 49))
    val_ds   = tf.data.Dataset.from_generator(lambda: data_handler.val_data_generator(widx=5, np_round_idxs=np_val_round_idxs), output_types=out_types, output_shapes=val_out_shape)
    # model arch
    unet_instance = unet_model(name='fr predictor', **config)
    model = unet_instance.UNet()
    model.summary()
    #train specs
    outmodelfile = config['training_configs']['outmodelfile']
    outhistfile  = config['training_configs']['outhistfile']
    loss_fn_txt      = config['training_configs']['loss_fn']
    if loss_fn_txt == 'weighted_cce':
        loss_fn = weighted_cce(CCE_WEIGHTS)
    else:
        loss_fn = loss_fn_txt
    check_point_callback = tf.keras.callbacks.ModelCheckpoint(filepath=outmodelfile, verbose=1, save_freq='epoch')
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy']) #binary_crossentropy

    # start training
    num_epochs = int(config['training_configs']['num_epochs'])
    hist = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[check_point_callback], verbose=1)
    with open(outhistfile, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
    model.save(outmodelfile)

if __name__ == '__main__':
   main(sys.argv)
