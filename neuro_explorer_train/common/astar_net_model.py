import os
import sys
import tensorflow as tf
from tensorflow.keras import layers

# This code has been adapted from the UNet model suggested in monodepth estimation
# C. Godard et al. "Unsupervised Monocular Depth Estimation with Left-Right Consistency," CVPR 17
# see "https://github.com/mrharicot/monodepth" for future details

class downscale_block(layers.Layer):
    def __init__(
            self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.pool = layers.MaxPool2D((2, 2), (2, 2))
    def call(self, input_tensor):
        d = self.convA(input_tensor)
        x = self.bn2a(d)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)
        x += d
        p = self.pool(x)
        return x, p

class upscale_block(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs):
        super().__init__(**kwargs)
        self.us = layers.UpSampling2D((2, 2))
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conc = layers.Concatenate()
    def call(self, x, skip):
        x = self.us(x)
        concat = self.conc([x, skip])
        x = self.convA(concat)
        x = self.bn2a(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)
        return x

class bottleneck_block(layers.Layer):
    def __init__(
            self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
    def call(self, x):
        x = self.convA(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.reluB(x)
        return x

class astarnet_model(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        #self.name = name
        self.kwargs = kwargs
        self.gmheight_p         = int(self.kwargs['data_configs']['gmheight_p'] )
        self.gmwidth_p          = int(self.kwargs['data_configs']['gmwidth_p'] )
        # network configs
        self.output_channels    = int(self.kwargs['network_configs']['output_channels'])
        self.input_channels     = int(self.kwargs['network_configs']['input_channels'])
        self.batch_size         = int(self.kwargs['network_configs']['batch_size'])
        self.num_filters   = int(self.kwargs['network_configs']['num_filters'])
        self.batch_norm         = bool(self.kwargs['network_configs']['batch_norm'])
        self.loss_weights    = self.kwargs['network_configs']['loss_weights']
        self.first_loss_weight  = self.loss_weights[0] #0.1 #0.85 #(default)
#        self.ssim_loss_weight   = self.loss_weights[1] #0.9 # 0.1  #(default)
#        self.edge_loss_weight = self.loss_weights[2] #0.9 #0.9 # edge loss generates artifacts

        #self.threshold        = 1.0
        #print("loss weights: %f %f %f \n"%( self.l1_loss_weight, self.ssim_loss_weight, self.edge_loss_weight ))
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.image_shape = [self.gmheight_p, self.gmwidth_p, self.input_channels]
        nf = self.num_filters
        f = [nf, nf*2, nf*4, nf*8, nf*16]
        self.downscale_blocks = [
            downscale_block(f[0]),
            downscale_block(f[1]),
            downscale_block(f[2]),
            downscale_block(f[3]),
        ]
        self.bottle_neck_block = bottleneck_block(f[4])
        self.upscale_blocks = [
            upscale_block(f[3]),
            upscale_block(f[2]),
            upscale_block(f[1]),
            upscale_block(f[0]),
        ]
        self.conv_layer = layers.Conv2D(1, (1, 1), padding="same", activation="tanh")
        inputs = tf.keras.layers.Input(shape=self.image_shape)
        x = inputs
        c1, p1 = self.downscale_blocks[0](x)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)
        bn = self.bottle_neck_block(p4)
        u1 = self.upscale_blocks[0](bn, c4)
        u2 = self.upscale_blocks[1](u1, c3)
        u3 = self.upscale_blocks[2](u2, c2)
        u4 = self.upscale_blocks[3](u3, c1)
        outputs = self.conv_layer(u4)
        self.model = tf.keras.Model(inputs, outputs)

    def compute_loss(self, target, pred):
        tf_loss = self.loss(y_true=target, y_pred=pred)
        return tf.reduce_mean(tf_loss)

    @property
    def metrics(self):
        return [self.loss_metric]
    def train_step(self, batch_data):
        input, target = batch_data
        with tf.GradientTape() as tape:
            pred = self(input, training=True)
            loss = self.compute_loss(target, pred) #self.calculate_loss(target, pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(target, pred)
        return {m.name: m.result() for m in self.metrics}
        # self.loss_metric.update_state(loss)
        # return {
        #     "loss": self.loss_metric.result(),
        # }
    def test_step(self, batch_data):
        input, target = batch_data
        pred = self(input, training=False)
        loss = self.compute_loss(target, pred) #self.calculate_loss(target, pred)
        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }
    def call(self, inputs):
        return self.model(inputs)