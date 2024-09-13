
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
import pickle
import time
base_dir = os.path.abspath('../../')
sys.path.insert(0, '%s/utils'%base_dir)
sys.path.insert(0, '%s/common'%base_dir)
from data_augmentation import rotate, flip
from data_generator import data_generator
from custom_loss_functions import switch_loss_fn
from custom_metrics import switch_metics

import yaml

def to_continous_label( gtimg_input, num_classes ): # gtimg uint8 w/ max 255
    gtimg = gtimg_input.squeeze()
    (h, w) = gtimg.shape
    gt_class = np.zeros([h, w, num_classes], dtype=np.float32)
    step_size = int( 256 / num_classes )
    m_bin = np.where(gtimg < step_size, 1, 0)
    gt_class[:, :, 0] = m_bin
    for chidx in range(1, num_classes):
        m_floor = np.where(gtimg >= chidx * step_size, 1, 0)
        m_ceil  = np.where(gtimg < (chidx+1) * step_size, 1, 0)
        m_bin = np.multiply(m_ceil, m_floor)
        gt_class[:, :, chidx] = m_bin.astype(np.float32)
    return gt_class
def binlabl_to_bimg(blabel, num_classes ):
    bimg = np.where(blabel > 0.5, 1, 0)
    return bimg
def labels_to_bimg( in_class, num_classes ): # gtimg uint8 w/ max 255
    #assert(in_class.ndim == 3)
    bimg = tf.argmax(in_class, axis=-1) / (num_classes-1) * 255
    return tf.cast(bimg, tf.uint8)

from tensorflow.keras.losses import Loss
class weighted_cce(Loss):
    def __init__(self, weights):
        super().__init__()
        self.class_weights = weights #[x / sum(weights) for x in weights]
    def call(self, y_true, y_pred):
        (B, H, W, C) = y_true.shape
        tf_y = y_true
        tf_x = y_pred
        #tf_weights = self.class_weights
        tf_weighted_true = tf.multiply(tf_y, tf.constant(self.class_weights))
        tf_weights = tf.reduce_sum(tf_weighted_true, axis=-1)
        tf_loss = K.categorical_crossentropy(y_true, y_pred)
        # print( "tf_y shape ", tf_y.shape )
        # print( "tf_weight shape: ", tf_weights )
        # print( "tf loss shape: ", tf_loss.shape)
        tf_loss_weighted = tf.multiply(tf_loss, tf_weights[..., 0])
        return tf_loss_weighted

from sklearn.metrics import mean_squared_error
class CustomRRMSE( tf.keras.metrics.Metric ):
    def __init__(self, name='custom_rrmse', **kwargs):
        super().__init__(name=name, **kwargs)
        #self.threshold = threshold
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.output_channels    = int(kwargs['network_configs']['output_channels'])
    def true_fn(self, rrmse):
        self.total.assign_add(rrmse)
        self.count.assign_add(1.)
    def false_fn(self):  # dummy fn
        self.total.assign_add(0.)
        self.count.assign_add(0.)
    def update_state(self, y_true, y_pred, sample_weight=None):
        gimg = labels_to_bimg(y_true[0, :, :, :], self.output_channels) # 0 ~ 255 intensity
        bimg = labels_to_bimg(y_pred[0, :, :, :], self.output_channels) # 0 ~ 255
        gimg_f = tf.cast(gimg, dtype=tf.float32)
        bimg_f = tf.cast(bimg, dtype=tf.float32)
        val_cells = tf.cast(tf.where(gimg_f > 0, 1, 0), dtype=tf.float32)
        gimg_f = gimg_f * val_cells
        bimg_f = bimg_f * val_cells
        nz_cnt = tf.math.reduce_sum(val_cells)
        sqe = tf.math.reduce_sum( (gimg_f - bimg_f) * (gimg_f - bimg_f) )
        mse = sqe / nz_cnt
        pred_sq_sum = tf.math.reduce_sum( bimg_f * bimg_f )
        rrmse = (mse / pred_sq_sum) ** 0.5
        cond1 = pred_sq_sum >= 1
        cond2 = nz_cnt > 0
        cond = tf.logical_and(cond1, cond2)
        return tf.cond(cond, lambda: self.true_fn(rrmse), lambda: self.false_fn() )
    def result(self):
        return self.total / self.count

class CustomRMSE( tf.keras.metrics.Metric ):
    def __init__(self, name='custom_rmse', **kwargs):
        super().__init__(name=name, **kwargs)
        #self.threshold = threshold
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    def true_fn(self, rmse):
        self.total.assign_add(rmse)
        self.count.assign_add(1.)
    def false_fn(self):  # dummy fn
        self.total.assign_add(0.)
        self.count.assign_add(0.)
    def update_state(self, y_true, y_pred, sample_weight=None):
        gimg = labels_to_bimg(y_true[0, :, :, :], self.output_channels) # 0 ~ 255 intensity
        bimg = labels_to_bimg(y_pred[0, :, :, :], self.output_channels) # 0 ~ 255
        gimg_f = tf.cast(gimg, dtype=tf.float32)
        bimg_f = tf.cast(bimg, dtype=tf.float32)
        val_cells = tf.cast( tf.where(gimg_f > 0, 1, 0), dtype=tf.float32)
        gimg_f = gimg_f * val_cells
        bimg_f = bimg_f * val_cells
        nz_cnt = tf.math.reduce_sum(val_cells)
        sqe = tf.math.reduce_sum( (gimg_f - bimg_f) * (gimg_f - bimg_f) )
        mse = sqe / nz_cnt
        rmse = mse ** 0.5
        cond = nz_cnt > 0
        return tf.cond(cond, lambda: self.true_fn(rmse), lambda: self.false_fn() )
    def result(self):
        return self.total / self.count

#def custom_nonzero_accuracy()

def main(argv):
    #with open('/home/hankm/results/neuro_exploration_res/astar_net/cont_resolution/params/astarnet_params_cont.yaml') as f:
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_handler = data_generator(name='data_handler', **config)

    cce_weights = np.ones([data_handler.output_channels], dtype='float32')  # * 3.
    cce_weights[0] = .1
    map_weights = np.ones([data_handler.gmheight_p, data_handler.gmwidth_p, data_handler.output_channels], dtype=np.float32)
    for ii in range(0, data_handler.gmheight_p):
        for jj in range(0, data_handler.gmwidth_p):
            map_weights[ii, jj, :] = cce_weights
    #train specs
    modelfile = config['training_configs']['outmodelfile']
    histfile  = config['training_configs']['outhistfile']
    loss_fn_txt      = config['training_configs']['loss_fn']
    loss_fn = switch_loss_fn(loss_fn_txt)

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
    model.compile(optimizer='adam', loss=loss_fn, metrics=metric_fn)

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
        #bimg = pi[0, ...]
        #gimg = gi[0, ...]
        plt.subplot(131)
        plt.imshow(im.squeeze())
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
        cv2.imwrite('%s/input%04d.png'%(outpath, sidx), im.squeeze() * 255.)

if __name__ == '__main__':
   main(sys.argv)