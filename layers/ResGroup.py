__author__ = 'Titi'

import tensorflow as tf
import numpy as np

class resGroup():
    def __init__(self, n_blocks, input_channel):
        self.n_blocks = n_blocks
        self.input_channel = input_channel
        
    def __call__(self, input_feat):
        blocks = [None for _ in range(self.n_blocks)]
        for n in range(self.n_blocks):
            if n == 0:
                blocks[n] = resBlock(self.input_channel, n)
                self.tm_block = blocks[n](input_feat)
            else:
                blocks[n] = resBlock(self.input_channel, n)
                self.tm_block = blocks[n](self.tm_block)
        
        group_return = tf.math.add(input_feat, self.tm_block)
        
        return group_return

class resBlock():
    def __init__(self, input_channel, n):
        self.channel = input_channel
        self.n = n
        
    def __call__(self, input_feat):
        print('Number {} resBlock'.format(self.n+1))
        with tf.variable_scope('resblock'+str(self.n+1), reuse=tf.AUTO_REUSE):
            tm_block = tf.layers.conv2d(inputs=input_feat,
                                        filters=self.channel,
                                        kernel_size=3,
                                        strides=1,
                                        padding='same',
                                        name='resBlock_conv_1')
            tm_block = tf.nn.relu(tm_block)
            tm_block = tf.layers.conv2d(inputs=tm_block,
                                        filters=self.channel,
                                        kernel_size=3,
                                        strides=1,
                                        padding='same',
                                        name='resBlock_conv_2')

            # CA module
            tm_ca = tf.reduce_mean(tm_block, [1, 2], keep_dims=True)
            tm_ca = tf.layers.conv2d(inputs=tm_ca,
                                     filters=self.channel // 2,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name='resCA_conv_1')
            tm_ca = tf.nn.relu(tm_ca)
            tm_ca = tf.layers.conv2d(inputs=tm_ca,
                                     filters=self.channel,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name='resCA_conv_2')
            tm_ca_weight = tf.sigmoid(tm_ca)
            ca_return = tf.math.multiply(tm_block, tm_ca_weight)

            block_return = tf.math.add(input_feat, ca_return)

            return block_return