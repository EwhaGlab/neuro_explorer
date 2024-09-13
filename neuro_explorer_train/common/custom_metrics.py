
import sys
import os
import keras.metrics

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
base_dir = os.path.abspath('../../')
sys.path.insert(0, '%s/utils'%base_dir)
sys.path.insert(0, '%s/common'%base_dir)

def labels_to_bimg( in_class, num_classes ): # gtimg uint8 w/ max 255
    #assert(in_class.ndim == 3)
    bimg = tf.argmax(in_class, axis=-1) / (num_classes-1) * 255
    return tf.cast(bimg, tf.uint8)

from sklearn.metrics import mean_squared_error
class weighted_rrmse( tf.keras.metrics.Metric ):
    def __init__(self, name='weighted_rrmse', output_channels=1, **kwargs):
        super().__init__(name=name, **kwargs)
        #self.threshold = threshold
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.output_channels = output_channels
    def true_fn(self, rrmse):
        self.total.assign_add(rrmse)
        self.count.assign_add(1.)
    def false_fn(self):  # dummy fn
        self.total.assign_add(0.)
        self.count.assign_add(0.)
    def update_state(self, y_true, y_pred, sample_weight=None):
        #gimg = labels_to_bimg(y_true[0, :, :, :], self.output_channels) # 0 ~ 255 intensity
        #bimg = labels_to_bimg(y_pred[0, :, :, :], self.output_channels) # 0 ~ 255
        gimg_f = tf.cast(y_true, dtype=tf.float32)
        bimg_f = tf.cast(y_pred, dtype=tf.float32)
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

class rrmse( tf.keras.metrics.Metric ):
    def __init__(self, name='rrmse', output_channels=1, **kwargs):
        super().__init__(name=name, **kwargs)
        #self.threshold = threshold
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.output_channels = output_channels
    def true_fn(self, rrmse):
        self.total.assign_add(rrmse)
        self.count.assign_add(1.)
    def false_fn(self):  # dummy fn
        self.total.assign_add(0.)
        self.count.assign_add(0.)
    def update_state(self, y_true, y_pred, sample_weight=None):
        gimg = y_true #labels_to_bimg(y_true[0, :, :, :], self.output_channels) # 0 ~ 255 intensity
        bimg = y_pred #labels_to_bimg(y_pred[0, :, :, :], self.output_channels) # 0 ~ 255
        gimg_f = tf.cast(gimg, dtype=tf.float32)
        bimg_f = tf.cast(bimg, dtype=tf.float32)
        val_cells = tf.cast(tf.where(gimg_f > 0, 1, 0), dtype=tf.float32)
        gimg_f = gimg_f * val_cells
        bimg_f = bimg_f * val_cells
        nz_cnt = tf.math.reduce_sum(val_cells)
        sqe = tf.math.reduce_sum( (gimg_f - bimg_f) * (gimg_f - bimg_f) )
        mse = sqe
        pred_sq_sum = tf.math.reduce_sum( bimg_f * bimg_f )
        rrmse = (mse / pred_sq_sum) ** 0.5
        cond1 = pred_sq_sum >= 1
        cond2 = nz_cnt > 0
        cond = tf.logical_and(cond1, cond2)
        return tf.cond(cond, lambda: self.true_fn(rrmse), lambda: self.false_fn() )
    def result(self):
        return self.total / self.count

class weighted_rmse( tf.keras.metrics.Metric ):
    def __init__(self, name='custom_rmse', output_channels=1, **kwargs):
        super().__init__(name=name, **kwargs)
        #self.threshold = threshold
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.output_channels = output_channels
    def true_fn(self, rmse):
        self.total.assign_add(rmse)
        self.count.assign_add(1.)
    def false_fn(self):  # dummy fn
        self.total.assign_add(0.)
        self.count.assign_add(0.)
    def update_state(self, y_true, y_pred):
        gimg = y_true #labels_to_bimg(y_true[0, :, :, :], self.output_channels) # 0 ~ 255 intensity
        bimg = y_pred #labels_to_bimg(y_pred[0, :, :, :], self.output_channels) # 0 ~ 255
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

class maxpot_euc_dist( tf.keras.metrics.Metric ):
    def __init__(self, name='max_pot_euc_dist', output_channels=1, **kwargs):
        super().__init__(name=name, **kwargs)
        #self.threshold = threshold
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.output_channels = output_channels
    def true_fn(self, error):
        self.total.assign_add(error)
        self.count.assign_add(1.)
    def false_fn(self):  # dummy fn
        self.total.assign_add(0.)
        self.count.assign_add(0.)
    def update_state(self, y_true, y_pred, sample_weight=None):
        true_max = tf.reduce_max(y_true)
        pred_max = tf.reduce_max(y_pred)
        true_max_loc = tf.where(y_true == true_max)
        pred_max_loc = tf.where(y_pred == pred_max)
        true_max_loc = tf.cast(true_max_loc, dtype='float32')
        pred_max_loc = tf.cast(pred_max_loc, dtype='float32')
        dist_error = tf.norm( true_max_loc[0] - pred_max_loc[0], ord='euclidean')
        cond = pred_max > 0.2
        return tf.cond(cond, lambda: self.true_fn(dist_error), lambda: self.false_fn())
    def result(self):
        return self.total / self.count
class rel_potdiff( tf.keras.metrics.Metric ):
# We cannot make this function work b/c  pred_max location could be 0 on y_true
    def __init__(self, name='rel_potdiff', output_channels=1, **kwargs):
        super().__init__(name=name, **kwargs)
        #self.threshold = threshold
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.output_channels = output_channels
    def true_fn(self, potdiff):
        self.total.assign_add(potdiff)
        self.count.assign_add(1.)
    def false_fn(self):  # dummy fn
        self.total.assign_add(0.)
        self.count.assign_add(0.)
    def update_state(self, y_true, y_pred, sample_weight=None):
        true_max = tf.reduce_max(y_true)
        pred_max = tf.reduce_max(y_pred)
        pred_max_loc = tf.where(y_pred == pred_max)
        r = pred_max_loc[0][0]
        c = pred_max_loc[0][1]
        y_true_sq = tf.squeeze(y_true)
        est_max = y_true_sq[r, c]
        print(est_max)
        potdiff = tf.abs(true_max - est_max)
        cond1 = pred_max > 0.2
        #cond2 = est_max > 0.0  this is not guaranteed... so
        cond = cond1 #tf.logical_and(cond1, cond2)
        return tf.cond(cond, lambda: self.true_fn(potdiff), lambda: self.false_fn())
    def result(self):
        return self.total / self.count
def switch_metics(key):
    if key == "rrmse":
        metric_fn_instance = rrmse()
        print(f'*********************************************\n metric is RRMSE \n*********************************************\n ')
    elif key == "weighted_rrmse":
        metric_fn_instance = weighted_rrmse()
        print(f'*********************************************\n metric fn is weighted RRMSE \n*********************************************\n')
    elif key == 'maxpot_euc_dist':
        metric_fn_instance = maxpot_euc_dist()
        print(f'*********************************************\n metric is max pot euc dist \n*********************************************\n ')
    elif key == 'rel_potdiff':
        metric_fn_instance = rel_potdiff()
        print(f'*********************************************\n metric is relative pot diff\n*********************************************\n ')
    elif key == "mae":
        metric_fn_instance = tf.keras.metrics.MeanAbsoluteError()
        print(f'*********************************************\n metric fn is MAE \n*********************************************\n')
    elif key == "mse":
        metric_fn_instance = tf.keras.metrics.MeanSquaredError()
        print(f'*********************************************\n metric fn is MSE \n*********************************************\n')
    elif key == "rmse":
        metric_fn_instance = tf.keras.metrics.RootMeanSquaredError()
        print(f'*********************************************\n metric fn is RMSE \n*********************************************\n')
    else:
        _metric_fn = tf.keras.metrics.get(key)
        metric_fn_instance = _metric_fn()
        print(f'*********************************************\n loss fn is  {_metric_fn}.\n*********************************************\n')
    return metric_fn_instance