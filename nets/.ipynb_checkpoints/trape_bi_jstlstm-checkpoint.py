__author__ = 'jaden'

import tensorflow as tf
from layers.GradientHighwayUnit import GHU as ghu
from layers.B_jstLSTMCell import B_jstlstmCell as bjstlstm
import os.path
import numpy as np
import cv2
from utils import preprocess
# from gdl import GDL


def rnn(images, images_bw, mask_true, num_layers, num_hidden, filter_size, stride=1,
        seq_length=20, input_length=5, tln=True):
    ###'num_hidden', '32,16,16,16', 4 layers

    # inp_images = []
    gen_images = []
    lstm_fw = []
    lstm_bw = []
    lstm_bi = []
    cell_fw = []
    cell_bw = []
    cell_bi = []
    hidden_fw = []
    hidden_bw = []
    hidden_bi = []
    # shapeConcat = tf.concat([images, mask_true],axis=-1)
    # shape = shapeConcat.get_shape().as_list()
    # output_channels = shape[-1]/2
    shape = images.get_shape().as_list()
    output_channels = shape[-1]
    # Time Machine
    tm_hidden_fw = [[None for i in range(seq_length-2)] for k in range(4)]
    tm_hidden_bw = [[None for i in range(seq_length-2)] for k in range(4)]
    tm_mem_fw = [[None for i in range(seq_length-2)] for k in range(4)]
    tm_mem_bw = [[None for i in range(seq_length-2)] for k in range(4)]
    # loss_gdl = GDL()


    for i in xrange(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]  ### [4-1]=3 equels [i-1]=-1 equels 64
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell_fw = bjstlstm('lstm_fw_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm_fw.append(new_cell_fw)
        cell_fw.append(None)
        hidden_fw.append(None)

    for i in xrange(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]  ### [4-1]=3 equels [i-1]=-1 equels 64
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell_bw = bjstlstm('lstm_bw_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm_bw.append(new_cell_bw)
        cell_bw.append(None)
        hidden_bw.append(None)

    # for i in xrange(num_layers-2, num_layers-1):
    #     num_hidden_in = num_hidden[i-1]
    #     new_bicell = bjstlstm('blstm_'+str(i+1),
    #                       filter_size,
    #                       num_hidden_in,
    #                       num_hidden[i],
    #                       shape,
    #                       tln=tln)
    #     lstm_bi.append(new_bicell)
    #     cell_bi.append(None)
    #     hidden_bi.append(None)

    gradient_highway_fw = ghu('highway_fw', filter_size, num_hidden[0], tln=tln)
    gradient_highway_bw = ghu('highway_bw', filter_size, num_hidden[0], tln=tln)

    mem_fw = None
    z_t_fw = None
    mem_bw = None
    z_t_bw = None

    for t_layer1 in xrange(seq_length-2):   ### t_layer1 = time
        with tf.variable_scope('b_jstlstm_l1', reuse=tf.AUTO_REUSE):
            inputs_fw = mask_true[:,t_layer1]*images[:,t_layer1] + (1-mask_true[:,t_layer1])*sample_Z((1-mask_true[:,t_layer1]))
            # inputs_fwConcat = tf.concat([inputs_fw, mask_true[:,t_layer1]],axis=-1)
            tf.summary.image('masktrue_fw', reshape_patch_back_gen(mask_true[:,t_layer1], 4), 29)
            tf.summary.image('input_fw', reshape_patch_back_gen(inputs_fw, 4), 29)

            hidden_fw[0], cell_fw[0], mem_fw = lstm_fw[0](inputs_fw, hidden_fw[0], cell_fw[0], mem_fw)

            z_t_fw = gradient_highway_fw(hidden_fw[0], z_t_fw)
            
            tm_hidden_fw[0][t_layer1] = z_t_fw
            tm_mem_fw[0][t_layer1] = mem_fw

        with tf.variable_scope('b_jstlstm_l1', reuse=tf.AUTO_REUSE):
            inputs_bw = mask_true[:,seq_length-1-t_layer1]*images_bw[:,t_layer1] + (1-mask_true[:,seq_length-1-t_layer1])*sample_Z((1-mask_true[:,seq_length-1-t_layer1]))
            
            hidden_bw[0], cell_bw[0], mem_bw = lstm_bw[0](inputs_bw, hidden_bw[0], cell_bw[0], mem_bw)
            z_t_bw = gradient_highway_bw(hidden_bw[0], z_t_bw)

            tm_hidden_bw[0][t_layer1] = z_t_bw
            tm_mem_bw[0][t_layer1] = mem_bw


    hiddenConcatConv_l2 = [None for i in range(seq_length-4)]
    memConcatConv_l2 = [None for i in range(seq_length-4)]
    for t_layer2 in xrange(seq_length-4):   ### t_layer2 = time
        with tf.variable_scope('merge_l2', reuse=tf.AUTO_REUSE):
            if t_layer2 < int((seq_length-4)/2):
                hiddenConcat_bw = tf.concat([tm_hidden_bw[0][t_layer2], tm_hidden_fw[0][-1-t_layer2]], axis=-1)
                hiddenConcatConv_l2[-1-t_layer2] = tf.layers.conv2d(inputs=hiddenConcat_bw,
                                            filters=tm_hidden_fw[0][t_layer2].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="B_h_merge_l2")
                memConcat_bw = tf.concat([tm_mem_bw[0][t_layer2], tm_mem_fw[0][-1-t_layer2]], axis=-1)
                memConcatConv_l2[-1-t_layer2] = tf.layers.conv2d(inputs=memConcat_bw,
                                            filters=tm_mem_fw[0][t_layer2].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="B_m_merge_l2")
            
                hiddenConcat_fw = tf.concat([tm_hidden_fw[0][t_layer2], tm_hidden_bw[0][-1-t_layer2]],axis=-1)
                hiddenConcatConv_l2[t_layer2] = tf.layers.conv2d(inputs=hiddenConcat_fw,
                                            filters=tm_hidden_fw[0][t_layer2].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="F_h_merge_l2")
                memConcat_fw = tf.concat([tm_mem_fw[0][t_layer2], tm_mem_bw[0][-1-t_layer2]],axis=-1)
                memConcatConv_l2[t_layer2] = tf.layers.conv2d(inputs=memConcat_fw,
                                            filters=tm_mem_fw[0][t_layer2].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="F_m_merge_l2")
        
        with tf.variable_scope('b_jstlstm_l2', reuse=tf.AUTO_REUSE):
            hidden_bw[1], cell_bw[1], mem_bw = lstm_bw[1](hiddenConcatConv_l2[-1-t_layer2], hidden_bw[1], cell_bw[1], memConcatConv_l2[-1-t_layer2])

            tm_hidden_bw[1][t_layer2+1] = hidden_bw[1]
            tm_mem_bw[1][t_layer2+1] = mem_bw

        with tf.variable_scope('b_jstlstm_l2', reuse=tf.AUTO_REUSE):
            hidden_fw[1], cell_fw[1], mem_fw = lstm_fw[1](hiddenConcatConv_l2[t_layer2], hidden_fw[1], cell_fw[1], memConcatConv_l2[t_layer2])
            
            tm_hidden_fw[1][t_layer2+1] = hidden_fw[1]
            tm_mem_fw[1][t_layer2+1] = mem_fw


    hiddenConcatConv_l3 = [None for i in range(seq_length-6)]
    memConcatConv_l3 = [None for i in range(seq_length-6)]
    for t_layer3 in xrange(seq_length-6):   ### t_layer3 = time
        with tf.variable_scope('merge_l3', reuse=tf.AUTO_REUSE):
            if t_layer3 < int((seq_length-6)/2):
                hiddenConcat_fw = tf.concat([tm_hidden_fw[1][(t_layer3+1)], tm_hidden_bw[1][-1-(t_layer3+1)]],axis=-1)
                hiddenConcatConv_l3[t_layer3] = tf.layers.conv2d(inputs=hiddenConcat_fw,
                                            filters=tm_hidden_fw[1][(t_layer3+1)].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="F_h_merge_l3")
                memConcat_fw = tf.concat([tm_mem_fw[1][(t_layer3+1)], tm_mem_bw[1][-1-(t_layer3+1)]],axis=-1)
                memConcatConv_l3[t_layer3] = tf.layers.conv2d(inputs=memConcat_fw,
                                            filters=tm_mem_fw[1][(t_layer3+1)].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="F_m_merge_l3")

                hiddenConcat_bw = tf.concat([tm_hidden_bw[1][(t_layer3+1)], tm_hidden_fw[1][-1-(t_layer3+1)]], axis=-1)
                hiddenConcatConv_l3[-1-t_layer3] = tf.layers.conv2d(inputs=hiddenConcat_bw,
                                            filters=tm_hidden_fw[1][(t_layer3+1)].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="B_h_merge_l3")
                memConcat_bw = tf.concat([tm_mem_bw[1][(t_layer3+1)], tm_mem_fw[1][-1-(t_layer3+1)]], axis=-1)
                memConcatConv_l3[-1-t_layer3] = tf.layers.conv2d(inputs=memConcat_bw,
                                            filters=tm_mem_fw[1][(t_layer3+1)].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="B_m_merge_l3")

        with tf.variable_scope('b_jstlstm_l3', reuse=tf.AUTO_REUSE):
            hidden_fw[2], cell_fw[2], mem_fw = lstm_fw[2](hiddenConcatConv_l3[t_layer3], hidden_fw[2], cell_fw[2], memConcatConv_l3[t_layer3])

            tm_hidden_fw[2][t_layer3+2] = hidden_fw[2]
            tm_mem_fw[2][t_layer3+2] = mem_fw

        with tf.variable_scope('b_jstlstm_l3', reuse=tf.AUTO_REUSE):
            
            hidden_bw[2], cell_bw[2], mem_bw = lstm_bw[2](hiddenConcatConv_l3[-1-t_layer3], hidden_bw[2], cell_bw[2], memConcatConv_l3[-1-t_layer3])

            tm_hidden_bw[2][t_layer3+2] = hidden_bw[2]
            tm_mem_bw[2][t_layer3+2] = mem_bw


    hiddenConcatConv_l4 = [None for i in range(seq_length-8)]
    memConcatConv_l4 = [None for i in range(seq_length-8)]
    hiddenConcatConv = [None for i in range(seq_length-8)]
    for t_layer4 in xrange(seq_length-8):   ### t_layer4 = time
        with tf.variable_scope('merge_l4', reuse=tf.AUTO_REUSE):
            if t_layer4 < int((seq_length-8)/2):
                hiddenConcat_bw = tf.concat([tm_hidden_bw[2][(t_layer4+2)], tm_hidden_fw[2][-1-(t_layer4+2)]], axis=-1)
                hiddenConcatConv_l4[-1-t_layer4] = tf.layers.conv2d(inputs=hiddenConcat_bw,
                                            filters=tm_hidden_fw[2][(t_layer4+2)].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="B_h_merge_l4")
                memConcat_bw = tf.concat([tm_mem_bw[2][(t_layer4+2)], tm_mem_fw[2][-1-(t_layer4+2)]], axis=-1)
                memConcatConv_l4[-1-t_layer4] = tf.layers.conv2d(inputs=memConcat_bw,
                                            filters=tm_mem_fw[2][(t_layer4+2)].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="B_m_merge_l4")

                hiddenConcat_fw = tf.concat([tm_hidden_fw[2][(t_layer4+2)], tm_hidden_bw[2][-1-(t_layer4+2)]],axis=-1)
                hiddenConcatConv_l4[t_layer4] = tf.layers.conv2d(inputs=hiddenConcat_fw,
                                            filters=tm_hidden_fw[2][(t_layer4+2)].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="F_h_merge_l4")
                memConcat_fw = tf.concat([tm_mem_fw[2][(t_layer4+2)], tm_mem_bw[2][-1-(t_layer4+2)]],axis=-1)
                memConcatConv_l4[t_layer4] = tf.layers.conv2d(inputs=memConcat_fw,
                                            filters=tm_mem_fw[2][(t_layer4+2)].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="F_m_merge_l4")

        with tf.variable_scope('b_jstlstm_l4', reuse=tf.AUTO_REUSE):
            hidden_bw[3], cell_bw[3], mem_bw = lstm_bw[3](hiddenConcatConv_l4[-1-t_layer4], hidden_bw[3], cell_bw[3], memConcatConv_l4[-1-t_layer4])

            tm_hidden_bw[3][t_layer4+3] = hidden_bw[3]
            tm_mem_bw[3][t_layer4+3] = mem_bw

        with tf.variable_scope('b_jstlstm_l4', reuse=tf.AUTO_REUSE):
            hidden_fw[3], cell_fw[3], mem_fw = lstm_fw[3](hiddenConcatConv_l4[t_layer4], hidden_fw[3], cell_fw[3], memConcatConv_l4[t_layer4])

            tm_hidden_fw[3][t_layer4+3] = hidden_fw[3]
            tm_mem_fw[3][t_layer4+3] = mem_fw


    x_gen = [None for i in range(seq_length-8)]
    for t_bi in xrange(seq_length-8):   ### t_bi = time
        with tf.variable_scope('bi_merge', reuse=tf.AUTO_REUSE):
            if t_bi < int((seq_length-8)/2):
                hiddenConcat = tf.concat([tm_hidden_fw[3][(t_bi+3)], tm_hidden_bw[3][-1-(t_bi+3)]],axis=-1)
                hiddenConcatConv[t_bi] = tf.layers.conv2d(inputs=hiddenConcat,
                                            filters=tm_hidden_fw[3][(t_bi+3)].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="F_h_merge")
                hiddenConcat = tf.concat([tm_hidden_bw[3][(t_bi+3)], tm_hidden_fw[3][-1-(t_bi+3)]],axis=-1)
                hiddenConcatConv[-1-t_bi] = tf.layers.conv2d(inputs=hiddenConcat,
                                            filters=tm_hidden_fw[3][(t_bi+3)].get_shape()[-1],
                                            kernel_size=1, strides=1, padding='same', name="B_h_merge")

                x_gen[t_bi] = tf.layers.conv2d(inputs=hiddenConcatConv[t_bi],
                                        filters=output_channels,
                                        kernel_size=1, strides=1, padding='same', name="bi_back_to_pixel")
                x_gen[-1-t_bi] = tf.layers.conv2d(inputs=hiddenConcatConv[-1-t_bi],
                                        filters=output_channels,
                                        kernel_size=1, strides=1, padding='same', name="bi_back_to_pixel")
            gen_images.append(x_gen[t_bi])
            print("t_bi: %d" % t_bi)
            tf.summary.image('x_gen', reshape_patch_back_gen(x_gen[t_bi], 4), seq_length-2)


    
    # inp_images = tf.stack(inp_images)
    gen_images = tf.stack(gen_images)
    # # # gen_images_bw = tf.stack(gen_images_bw)

    # [batch_size, seq_length, height, width, channels]
    # inp_images = tf.transpose(inp_images, [1,0,2,3,4])
    gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    # # # gen_images_bw = tf.transpose(gen_images_bw, [1,0,2,3,4])

    # No 0, 29, from 1 to seq_length-2
    l2Loss = tf.nn.l2_loss(gen_images - images[:,4:-4])
    l1Loss = tf.losses.absolute_difference(gen_images, images[:,4:-4])
    #hbloss = tf.losses.huber_loss(images[:,4:-4], gen_images, delta=1.5)
    gdlLoss = cal_gdl(gen_images, images[:,4:-4])
    loss = l2Loss + l1Loss + gdlLoss
    tf.summary.scalar('l2_Loss', l2Loss)
    tf.summary.scalar('l1_Loss', l1Loss)
    #tf.summary.scalar('huber_Loss', hbloss)
    tf.summary.scalar('gdl_Loss', gdlLoss)
    tf.summary.scalar('loss', loss)

    return [gen_images, loss, images, images_bw]


def reshape_patch_back_gen(patch_tensor, patch_size=4):
    print(np.shape(patch_tensor))
    batch_size = np.shape(patch_tensor)[0]
    patch_height = np.shape(patch_tensor)[1]
    patch_width = np.shape(patch_tensor)[2]
    channels = np.shape(patch_tensor)[3]
    img_channels = channels / (patch_size*patch_size)
    a = tf.reshape(patch_tensor, [batch_size,
                                patch_height, patch_width,
                                patch_size, patch_size,
                                img_channels])
    b = tf.transpose(a, [0,1,3,2,4,5])
    img_tensor = tf.reshape(b, [batch_size,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor

def sample_Z(m):
    return tf.random_uniform(np.shape(m), minval=0.0, maxval=1.0, dtype=tf.float32)

def cal_gdl(predImg, target):
        """
        Gradient Difference Loss
        Image gradient difference loss as defined by Mathieu et al. (https://arxiv.org/abs/1511.05440).

        """
        alpha = 1

        # [batch_size, seq_length, height, width, channels]
        predImg_col_grad = tf.abs(predImg[:, :, :, :-1, :] - predImg[:, :, :, 1:, :])
        predImg_row_grad = tf.abs(predImg[:, :, 1:, :, :] - predImg[:, :, :-1, :, :])
        target_col_grad = tf.abs(target[:, :, :, :-1, :] - target[:, :, :, 1:, :])
        target_row_grad = tf.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
        col_grad_loss = tf.abs(predImg_col_grad - target_col_grad)
        row_grad_loss = tf.abs(predImg_row_grad - target_row_grad)

        #loss = col_grad_loss + row_grad_loss
        loss = tf.reduce_sum(col_grad_loss ** alpha) + tf.reduce_sum(row_grad_loss ** alpha)
        return loss