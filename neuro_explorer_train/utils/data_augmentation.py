import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models, layers, losses, activations, regularizers, metrics
from keras import backend as K
import seaborn as sns
import fnmatch
from numpy import loadtxt
import matplotlib.pyplot as plt
import pickle
import time
import logging
import random

def rotate(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Args:
    :param x: Input image
    :param y: GT image
    :return: Augmented (rotated) image
    """
    # concatenate x and y
    assert(x.ndim == 3)
    x_nch = x.shape[2]
    y_nch = y.shape[2]
    xy = tf.concat([x, y], axis=-1)
    # Rotate 0, 90, 180, 270
    xy = tf.image.rot90(xy, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    x_rot = xy[:,:,0:x_nch]
    y_rot = xy[:,:,x_nch:x_nch+y_nch]
    return x_rot, y_rot
def flip(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Args:
    :param x: Input image !!!
    :param y: GT image
    :return: Augmented (flipped) image
    """
    # concatenate x and y
    assert(x.ndim == 3)
    x_nch = x.shape[2]
    y_nch = y.shape[2]
    xy = tf.concat([x, y], axis=-1)
    xy = tf.image.random_flip_left_right(xy)
    xy = tf.image.random_flip_up_down(xy)
    x_flip = xy[:,:,0:x_nch]
    y_flip = xy[:,:,x_nch:x_nch+y_nch]
    return x_flip, y_flip
def augment_random(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    :param x: Input image
    :param y: GT image
    :return: Augmented image
    """
    x, y = rotate(x, y)
    x, y = flip(x, y)
    return x, y