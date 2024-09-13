
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
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
    np_test_round_idx = np.asarray(range(0, 100)) #100))
    # start = time.time()
    val_ds   = tf.data.Dataset.from_generator(lambda: data_handler.test_data_generator(widx=5, np_round_idxs=np_test_round_idx), output_types=out_types, output_shapes=out_shapes)
    model = tf.keras.models.load_model(modelfile, compile=False) #, compile=False)  # custom_objects={ 'weighted_cce': weighted_cce(CCE_WEIGHTS) })
    model.compile(optimizer='adam', loss='mae', metrics=metric_fn)

    # load hist
    with open(histfile, "rb") as file_pi:
        history = pickle.load(file_pi)

    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    # plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    outpath = config['training_configs']['outpath']
    for sidx in range(0, 50):
        im, gi = val_ds.skip(sidx).take(1).as_numpy_iterator().next()
        pi = model.predict(im)
        plt.title("input")
        plt.subplot(132)
        plt.imshow(gi.squeeze())
        plt.title("GT")
        plt.subplot(133)
        plt.imshow(pi.squeeze())
        plt.title("prediction")
        #plt.show()
        outfile = '%s/fig_res%04d.png' %(outpath, sidx)
        plt.savefig(outfile, bbox_inches='tight')
        cv2.imwrite('%s/pred%04d.png'%(outpath, sidx), pi.squeeze() * 255.)
        cv2.imwrite('%s/gimg%04d.png'%(outpath, sidx), gi.squeeze() * 255.)
        im_sqz = im.squeeze()
        im_dummy = np.concatenate([im_sqz, np.zeros([512,512,1], dtype=np.float32)], axis=-1)
        cv2.imwrite('%s/input%04d.png'%(outpath, sidx), im_dummy * 255.)

if __name__ == '__main__':
   main(sys.argv)