import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from keras import backend as K
base_dir = os.path.abspath('../../')
sys.path.insert(0, '%s/utils'%base_dir)
sys.path.insert(0, '%s/common'%base_dir)

class huber_loss(tf.keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        sqd_loss = tf.square(error) / 2
        lin_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, sqd_loss, lin_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold":self.threshold}
class berhu_loss(tf.keras.losses.Loss):
    def __init__(self, coeff=0.2, **kwargs):
        self.coeff = coeff
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        max_err = tf.reduce_max(error)
        c = self.coeff * max_err
        is_small_error = tf.abs(error) < c
        l1_loss = tf.abs(error)
        sq_loss = error * error / (2. * c) + c / 2.
        return tf.where(is_small_error, l1_loss, sq_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "coeff": self.coeff}
class g2_smoothness_loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        y_true_nz = tf.cast(tf.where(y_true > 0, 1, 0), tf.float32)
        num_nz = tf.cast(tf.reduce_sum(y_true_nz), tf.float32)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        d2y_pred, dyx_pred = tf.image.image_gradients(dy_pred)
        d2x_pred, dxy_pred = tf.image.image_gradients(dx_pred)
        smoothness = d2y_pred + dxy_pred + dyx_pred + d2x_pred
        weighted_smoothness = smoothness #* y_true_nz
        # potmap smoothness
        potmap_smoothness_loss = weighted_smoothness
        return potmap_smoothness_loss
class g1_smoothness_loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        y_true_nz = tf.cast(tf.where(y_true > 0, 1, 0), tf.float32)
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
        weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))
        # potmap smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y
        potmap_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))
        return potmap_smoothness_loss
class max_euc_dist(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        true_max = tf.reduce_max(y_true)
        pred_max = tf.reduce_max(y_pred)
        true_max_loc = tf.where(y_true == true_max)
        pred_max_loc = tf.where(y_pred == pred_max)
        true_max_loc = tf.cast(true_max_loc, dtype='float32')
        pred_max_loc = tf.cast(pred_max_loc, dtype='float32')
        loss = tf.norm( true_max_loc[0] - pred_max_loc[0], ord='euclidean')
        return loss
class my_joint_loss(tf.keras.losses.Loss):
    def __init__(self, alpha, beta, **kwargs):
        super().__init__(**kwargs)
        self.huber = huber_loss(1.0)
        self.g_smoothness_loss = g1_smoothness_loss()
        self.alpha = alpha
        self.beta  = beta
    def call(self, y_true, y_pred):
        huloss = self.huber(y_true, y_pred)
        smloss = self.g_smoothness_loss(y_true, y_pred)
        return huloss * self.alpha + smloss * self.beta
class weighted_cce(tf.keras.losses.Loss):
    def __init__(self, map_weights):
        super().__init__()
        #self.class_weights = weights #[x / sum(weights) for x in weights]
        self.map_weights = map_weights
    def call(self, y_true, y_pred):
        (B, H, W, C) = y_true.shape
        tf_y = y_true   # exclusive
        tf_x = y_pred
        tf_weighted_true = tf.multiply(tf_y, tf.constant(self.map_weights))
        tf_weighted_true_sum = tf.reduce_sum(tf_weighted_true, axis=-1) #/ self.cce_weights_sum   # 0.1 ~ 1
        tf_loss = K.categorical_crossentropy(y_true, y_pred)
        tf_loss_weighted = tf.multiply(tf_loss, tf_weighted_true_sum[..., 0])
        return tf_loss_weighted
def switch_loss_fn(key):
    # loss_fn = {"huber": "huber_loss", "berhu": "berhu_loss"}.get(key, key)
    # print(f'loss fn is  {loss_fn}.')
    if key == "huber":
        #_loss_fn = huber_loss
        loss_fn_instance = huber_loss(1.0)
        print(f'*********************************************\n loss fn is  huber \n*********************************************\n ')
    elif key == "berhu":
        #_loss_fn = getattr(custom_loss_functions, "berhu_loss")
        loss_fn_instance = berhu_loss(0.2)
        print(f'*********************************************\n loss fn is  berhu \n *********************************************\n')
    elif key == "my_joint_loss":
        # _loss_fn = getattr(custom_loss_functions, "my_joint_loss")
        loss_fn_instance = my_joint_loss(0.9, 0.1)
        print(f'*********************************************\n loss fn is  my joint loss fn \n *********************************************\n')
    elif key == "mse":
        _loss_fn = tf.keras.losses.MeanSquaredError
        loss_fn_instance = _loss_fn()
        print(f'*********************************************\n loss fn is  {_loss_fn}.\n *********************************************\n')
    elif key == "mae":
        _loss_fn = tf.keras.losses.MeanAbsoluteError
        loss_fn_instance = _loss_fn()
        print(f'*********************************************\n loss fn is  {_loss_fn}.\n *********************************************\n')
    elif key == "max_euc_dist":
        loss_fn_instance = max_euc_dist()
        print(f'*********************************************\n loss fn is max euc dist\n *********************************************\n')
    else:
        raise Exception("********** UNKNOWN LOSS function !!! **************\n'")
        # _loss_fn = tf.keras.losses.get(key)
        # loss_fn_instance = _loss_fn()
        #print(f'*********************************************\n loss fn is  {_loss_fn}.\n *********************************************\n')
    return loss_fn_instance