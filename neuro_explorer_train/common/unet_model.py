import sys
import os
import keras.metrics
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, losses, activations, regularizers, metrics
from keras import backend as K
sys.path.insert(0, '/home/hankm/python_ws/neuro_ae/utils')

# creates a standard unet instance
class unet_model():
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.gmheight_p         = int(self.kwargs['data_configs']['gmheight_p'] )
        self.gmwidth_p          = int(self.kwargs['data_configs']['gmwidth_p'] )
        # network configs
        self.output_channels    = int(self.kwargs['network_configs']['output_channels'])
        self.input_channels     = int(self.kwargs['network_configs']['input_channels'])
        self.batch_size         = int(self.kwargs['network_configs']['batch_size'])
        self.unet_num_filters   = int(self.kwargs['network_configs']['num_filters'])
        self.batch_norm         = bool(self.kwargs['network_configs']['batch_norm'])
    def conv2d_block(self, input_tensor, n_filters, kernel_size=3):
        '''
        Add 2 convolutional layers with the parameters
        '''
        x = input_tensor
        for i in range(2):
            x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                                        kernel_initializer='he_normal', activation='relu', padding='same')(x)
        #x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer='he_normal',activation='relu',padding='same')(x)
        return x
    def encoder_block(self, inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
        '''
        Add 2 convolutional blocks and then perform down sampling on output of convolutions
        '''
        f = self.conv2d_block(inputs, n_filters)
        p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(f)
        p = tf.keras.layers.Dropout(dropout)(p)
        return f, p
    def encoder(self, inputs):
        '''
        defines the encoder or downsampling path.
        '''
        f1, p1 = self.encoder_block(inputs, n_filters=self.unet_num_filters)
        f2, p2 = self.encoder_block(p1, n_filters=self.unet_num_filters*2)
        f3, p3 = self.encoder_block(p2, n_filters=self.unet_num_filters*4)
        f4, p4 = self.encoder_block(p3, n_filters=self.unet_num_filters*8)
        return p4, (f1, f2, f3, f4)
# Bottlenect
    def bottleneck(self, inputs):
        bottle_neck = self.conv2d_block(inputs, n_filters=self.unet_num_filters*16)
        return bottle_neck
# Decoder
    def decoder_block(self, inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
        '''
        defines the one decoder block of the UNet
        '''
        u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides, padding='same')(inputs)
        c = tf.keras.layers.concatenate([u, conv_output])
        c = tf.keras.layers.Dropout(dropout)(c)
        c = self.conv2d_block(c, n_filters)
        return c
    def decoder(self, inputs, convs, output_channels):
        '''
        Defines the decoder of the UNet chaining together 4 decoder blocks.
        '''
        f1, f2, f3, f4 = convs
        c6 = self.decoder_block(inputs, f4, n_filters=self.unet_num_filters*8, kernel_size=3, strides=2)
        c7 = self.decoder_block(c6, f3, n_filters=self.unet_num_filters*4, kernel_size=3, strides=2)
        c8 = self.decoder_block(c7, f2, n_filters=self.unet_num_filters*2, kernel_size=3, strides=2)
        c9 = self.decoder_block(c8, f1, n_filters=self.unet_num_filters, kernel_size=3, strides=2)
        if output_channels == 1: # Binary (very important !!!)
           activation = 'sigmoid' # Make sure to use sigmoid !!!
        else:
           activation = 'softmax'
        # in the case of categorical classification, the above commented func is right
        # Our problem is not mutual-exclusive classification... That is, FR could be opt FR
        #activation = 'sigmoid'
        outputs = tf.keras.layers.Conv2D(output_channels, 1, activation=activation)(c9)
        return outputs
    def UNet(self):
        '''
        Defines the UNet by connecting the encoder, bottleneck and decoder
        '''
        inputs = tf.keras.layers.Input(shape=(self.gmheight_p, self.gmheight_p, self.input_channels))
        if self.batch_norm is True:
            inputs_bn = tf.keras.layers.BatchNormalization()(inputs)
            encoder_output, convs = self.encoder(inputs_bn)
        else:
            encoder_output, convs = self.encoder(inputs)
        bottle_neck = self.bottleneck(encoder_output)
        out_decode = self.decoder(bottle_neck, convs, self.output_channels)
        outputs = out_decode #FlatOut(out_decode)
        model = tf.keras.Model(inputs, outputs)
        return model