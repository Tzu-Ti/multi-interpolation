__author__ = 'Titi'

import tensorflow as tf
from layers.CausalLSTMCell import CausalLSTMCell as cslstm
from layers.GradientHighwayUnit import GHU as ghu
import numpy as np

def rnn(images, images_bw, mask_true, num_layers, num_hidden, filter_size, stride=1, seq_length=11, input_length=5, tln=True):
    
    gen_images = []
    lstm_fw = []
    lstm_bw = []
    cell_fw = []
    cell_bw = []
    hidden_fw = []
    hidden_bw = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1]
    # Time Machine (put memory and hidden per layer)
    tm_hidden_fw = [[None for i in range(seq_length)] for k in range(4)]
    tm_hidden_bw = [[None for i in range(seq_length)] for k in range(4)]
    tm_mem_fw = [[None for i in range(seq_length)] for k in range(4)]
    tm_mem_bw = [[None for i in range(seq_length)] for k in range(4)]
    
    ## Create causal lstm unit
    # Create forward causal lstm unit
    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = cslstm('lstm_fw_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm_fw.append(new_cell)
        cell_fw.append(None)
        hidden_fw.append(None)
    # Create backward causal lstm unit
    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = cslstm('lstm_bw_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm_bw.append(new_cell)
        cell_bw.append(None)
        hidden_bw.append(None)
        
    ## Create GHU unit
    # Create forward and backward GHU unit
    gradient_highway_fw = ghu('highway_fw', filter_size, num_hidden[0], tln=tln)
    gradient_highway_bw = ghu('highway_bw', filter_size, num_hidden[0], tln=tln)

    ## Create lstm memory output and GHU output
    # Create forward memory output and GHU output
    mem_fw = None
    z_t_fw = None
    # Create backward memory output and GHU output
    mem_bw = None
    z_t_bw = None

    print("seq_length:{}".format(seq_length))
    
    # Layer 1
    for t in range(seq_length):
        print("Layer 1")
        print("t:{}".format(t))
        
#         reuse = bool(gen_images)
        # Layer 1 Forward 
        with tf.variable_scope('bi_cslstm_l1', reuse=tf.AUTO_REUSE):
            # accroding mask_true replace with random noise
            inputs_fw = mask_true[:, t]*images[:, t] + (1-mask_true[:, t])*sample_Z((1-mask_true[:, t]))
            
            tf.summary.image('masktrue_fw', reshape_patch_back_gen(mask_true[:,t], 1), 11)
            tf.summary.image('input_fw', reshape_patch_back_gen(inputs_fw, 1), 11)
            
            hidden_fw[0], cell_fw[0], mem_fw = lstm_fw[0](inputs_fw, hidden_fw[0], cell_fw[0], mem_fw)
            z_t_fw = gradient_highway_fw(hidden_fw[0], z_t_fw)
            
            tm_hidden_fw[0][t] = z_t_fw
            tm_mem_fw[0][t] = mem_fw
        # Layer 1 Backward
        with tf.variable_scope('bi_cslstm_l1', reuse=tf.AUTO_REUSE):
            # accroding mask_true replace with random noise
            inputs_bw = mask_true[:, seq_length-t-1]*images_bw[:, t] + (1-mask_true[:, seq_length-t-1])*sample_Z((1-mask_true[:, seq_length-t-1]))
            
            hidden_bw[0], cell_bw[0], mem_bw = lstm_bw[0](inputs_bw, hidden_bw[0], cell_bw[0], mem_bw)
            z_t_bw = gradient_highway_bw(hidden_bw[0], z_t_bw)
        
            tm_hidden_bw[0][t] = z_t_bw
            tm_mem_bw[0][t] = mem_bw
    
    # Layer 2 only have 5 lstm
    hiddenConcatConv_l2 = [None for i in range(seq_length//2)]
    memConcatConv_l2 = [None for i in range(seq_length//2)]
    for t in range(seq_length//2):
        print("Layer 2")
        print("t:{}".format(t))
        
        # Merge forward and backward output from layer 1
        with tf.variable_scope('merge_l2', reuse=tf.AUTO_REUSE):
            if t < (seq_length//2//2):
                hiddenConcat = tf.concat([tm_hidden_fw[0][t*2], tm_hidden_bw[0][(seq_length//2-t-1)*2]], axis=-1)
                hiddenConcatConv_l2[t] = tf.layers.conv2d(inputs=hiddenConcat,
                                                          filters=tm_hidden_fw[0][t].get_shape()[-1],
                                                          kernel_size=1,
                                                          strides=1,
                                                          padding='same',
                                                          name='F_h_merge_l2')
                memConcat = tf.concat([tm_mem_fw[0][t*2], tm_mem_bw[0][(seq_length//2-t-1)*2]], axis=-1)
                memConcatConv_l2[t] = tf.layers.conv2d(inputs=memConcat,
                                                       filters=tm_mem_fw[0][t].get_shape()[-1],
                                                       kernel_size=1,
                                                       strides=1,
                                                       padding='same',
                                                       name='F_m_merge_l2')
            else:
                hiddenConcat = tf.concat([tm_hidden_bw[0][(seq_length//2-t-1)*2], tm_hidden_fw[0][t*2]], axis=-1)
                hiddenConcatConv_l2[t] = tf.layers.conv2d(inputs=hiddenConcat,
                                                           filters=tm_hidden_fw[0][t].get_shape()[-1],
                                                           kernel_size=1,
                                                           strides=1,
                                                           padding='same',
                                                           name='B_h_merge_l2')
                memConcat = tf.concat([tm_mem_bw[0][(seq_length//2-t-1)*2], tm_mem_fw[0][t*2]], axis=-1)
                memConcatConv_l2[t] = tf.layers.conv2d(inputs=memConcat,
                                                       filters=tm_mem_fw[0][t].get_shape()[-1],
                                                       kernel_size=1,
                                                       strides=1,
                                                       padding='same',
                                                       name='B_m_merge_l2')
    
    for t in range(seq_length//2):
        # Layer 2 Forward
        with tf.variable_scope('bi_cslstm_l2', reuse=tf.AUTO_REUSE):
            hidden_fw[1], cell_fw[1], mem_fw = lstm_fw[1](hiddenConcatConv_l2[t], hidden_fw[1], cell_fw[1], memConcatConv_l2[t])
            
            tm_hidden_fw[1][t] = hidden_fw[1]
            tm_mem_fw[1][t] = mem_fw
        # Layer 2 Backward
        with tf.variable_scope('bi_cslstm_l2', reuse=tf.AUTO_REUSE):
            hidden_bw[1], cell_bw[1], mem_bw = lstm_bw[1](hiddenConcatConv_l2[seq_length//2-t-1], hidden_bw[1], cell_bw[1], memConcatConv_l2[seq_length//2-t-1])
            tm_hidden_bw[1][t] = hidden_bw[1]
            tm_mem_bw[1][t] = mem_bw
            
    # Layer 3 only have 5 lstm
    hiddenConcatConv_l3 = [None for i in range(seq_length//2)]
    memConcatConv_l3 = [None for i in range(seq_length//2)]
    for t in range(seq_length//2):
        print("Layer 3")
        print("t:{}".format(t))
        
        # Merge forward and backward output from layer 2
        with tf.variable_scope('merge_l3', reuse=tf.AUTO_REUSE):
            if t < (seq_length//2//2):
                hiddenConcat = tf.concat([tm_hidden_fw[1][t], tm_hidden_bw[1][seq_length//2-t-1]], axis=-1)
                hiddenConcatConv_l3[t] = tf.layers.conv2d(inputs=hiddenConcat,
                                                          filters=tm_hidden_fw[1][t].get_shape()[-1],
                                                          kernel_size=1,
                                                          strides=1,
                                                          padding='same',
                                                          name='F_h_merge_l3')
                memConcat = tf.concat([tm_mem_fw[1][t], tm_mem_bw[1][seq_length//2-t-1]], axis=-1)
                memConcatConv_l3[t] = tf.layers.conv2d(inputs=memConcat,
                                                       filters=tm_mem_fw[1][t].get_shape()[-1],
                                                       kernel_size=1,
                                                       strides=1,
                                                       padding='same',
                                                       name='F_m_merge_l3')
            else:
                hiddenConcat = tf.concat([tm_hidden_bw[1][seq_length//2-t-1], tm_hidden_fw[1][t]], axis=-1)
                hiddenConcatConv_l3[t] = tf.layers.conv2d(inputs=hiddenConcat,
                                                           filters=tm_hidden_fw[1][t].get_shape()[-1],
                                                           kernel_size=1,
                                                           strides=1,
                                                           padding='same',
                                                           name='B_h_merge_l3')
                memConcat = tf.concat([tm_mem_bw[1][seq_length//2-t-1], tm_mem_fw[1][t]], axis=-1)
                memConcatConv_l3[t] = tf.layers.conv2d(inputs=memConcat,
                                                       filters=tm_mem_fw[1][t].get_shape()[-1],
                                                       kernel_size=1,
                                                       strides=1,
                                                       padding='same',
                                                       name='B_m_merge_l3')
    
    for t in range(seq_length//2):
        # Layer 3 Forward
        with tf.variable_scope('bi_cslstm_l3', reuse=tf.AUTO_REUSE):
            hidden_fw[2], cell_fw[2], mem_fw = lstm_fw[2](hiddenConcatConv_l3[t], hidden_fw[2], cell_fw[2], memConcatConv_l3[t])
            
            tm_hidden_fw[2][t] = hidden_fw[2]
            tm_mem_fw[2][t] = mem_fw
        # Layer 3 Backward
        with tf.variable_scope('bi_cslstm_l3', reuse=tf.AUTO_REUSE):
            hidden_bw[2], cell_bw[2], mem_bw = lstm_bw[2](hiddenConcatConv_l3[seq_length//2-t-1], hidden_bw[2], cell_bw[2], memConcatConv_l3[seq_length//2-t-1])
            tm_hidden_bw[2][t] = hidden_bw[2]
            tm_mem_bw[2][t] = mem_bw
            
    # Layer 4 only have 5 lstm
    hiddenConcatConv_l4 = [None for i in range(seq_length//2)]
    memConcatConv_l4 = [None for i in range(seq_length//2)]
    for t in range(seq_length//2):
        print("Layer 4")
        print("t:{}".format(t))
        
        # Merge forward and backward output from layer 3
        with tf.variable_scope('merge_l4', reuse=tf.AUTO_REUSE):
            if t < (seq_length//2//2):
                hiddenConcat = tf.concat([tm_hidden_fw[2][t], tm_hidden_bw[2][seq_length//2-t-1]], axis=-1)
                hiddenConcatConv_l4[t] = tf.layers.conv2d(inputs=hiddenConcat,
                                                          filters=tm_hidden_fw[2][t].get_shape()[-1],
                                                          kernel_size=1,
                                                          strides=1,
                                                          padding='same',
                                                          name='F_h_merge_l4')
                memConcat = tf.concat([tm_mem_fw[2][t], tm_mem_bw[2][seq_length//2-t-1]], axis=-1)
                memConcatConv_l4[t] = tf.layers.conv2d(inputs=memConcat,
                                                       filters=tm_mem_fw[2][t].get_shape()[-1],
                                                       kernel_size=1,
                                                       strides=1,
                                                       padding='same',
                                                       name='F_m_merge_l4')
            else:
                hiddenConcat = tf.concat([tm_hidden_bw[2][seq_length//2-t-1], tm_hidden_fw[2][t]], axis=-1)
                hiddenConcatConv_l4[t] = tf.layers.conv2d(inputs=hiddenConcat,
                                                           filters=tm_hidden_fw[2][t].get_shape()[-1],
                                                           kernel_size=1,
                                                           strides=1,
                                                           padding='same',
                                                           name='B_h_merge_l4')
                memConcat = tf.concat([tm_mem_bw[2][seq_length//2-t-1], tm_mem_fw[2][t]], axis=-1)
                memConcatConv_l4[t] = tf.layers.conv2d(inputs=memConcat,
                                                       filters=tm_mem_fw[2][t].get_shape()[-1],
                                                       kernel_size=1,
                                                       strides=1,
                                                       padding='same',
                                                       name='B_m_merge_l4')
    
    for t in range(seq_length//2):
        # Layer 4 Forward
        with tf.variable_scope('bi_cslstm_l4', reuse=tf.AUTO_REUSE):
            hidden_fw[3], cell_fw[3], mem_fw = lstm_fw[3](hiddenConcatConv_l4[t], hidden_fw[3], cell_fw[3], memConcatConv_l4[t])
            
            tm_hidden_fw[3][t] = hidden_fw[3]
            tm_mem_fw[3][t] = mem_fw
        # Layer 4 Backward
        with tf.variable_scope('bi_cslstm_l4', reuse=tf.AUTO_REUSE):
            hidden_bw[3], cell_bw[3], mem_bw = lstm_bw[3](hiddenConcatConv_l4[seq_length//2-t-1], hidden_bw[3], cell_bw[3], memConcatConv_l4[seq_length//2-t-1])
            tm_hidden_bw[3][t] = hidden_bw[3]
            tm_mem_bw[3][t] = mem_bw
            
        
    
    # generate output image
    hiddenConcatConv = [None for i in range(seq_length//2)]
    x_gen = [None for i in range(seq_length//2)]
    for t in range(seq_length//2):
        with tf.variable_scope('bi_merge', reuse=tf.AUTO_REUSE):
            if t < (seq_length//2//2):
                hiddenConcat = tf.concat([tm_hidden_fw[3][t], tm_hidden_bw[3][seq_length//2-t-1]], axis=-1)
                hiddenConcatConv[t] = tf.layers.conv2d(inputs=hiddenConcat,
                                                       filters=tm_hidden_fw[3][t].get_shape()[-1],
                                                       kernel_size=1,
                                                       strides=1,
                                                       padding='same',
                                                       name='F_h_merge')
            else:
                hiddenConcat = tf.concat([tm_hidden_bw[3][seq_length//2-t-1], tm_hidden_fw[3][t]], axis=-1)
                hiddenConcatConv[t] = tf.layers.conv2d(inputs=hiddenConcat,
                                                       filters=tm_hidden_bw[3][t].get_shape()[-1],
                                                       kernel_size=1,
                                                       strides=1,
                                                       padding='same',
                                                       name='B_h_merge')
    for t in range(seq_length//2):
        with tf.variable_scope('generate', reuse=tf.AUTO_REUSE):
            x_gen[t] = tf.layers.conv2d(inputs=hiddenConcatConv[t],
                                        filters=output_channels,
                                        kernel_size=1,
                                        strides=1,
                                        padding='same',
                                        name='bi_back_to_pixel')
            gen_images.append(x_gen[t])
            print("generate t: %d" % t)
            tf.summary.image('x_gen', reshape_patch_back_gen(x_gen[t], 1), seq_length)

    gen_images = tf.stack(gen_images)
    # [batch_size, seq_length, height, width, channels]
    gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    
    gt_images = [images[:, i*2+1] for i in range(seq_length//2)]
    gt_images = tf.stack(gt_images)
    gt_images = tf.transpose(gt_images, [1,0,2,3,4])
    
    l2Loss = tf.nn.l2_loss(gen_images - gt_images)
    l1Loss = tf.losses.absolute_difference(gen_images, gt_images)
    gdlLoss = cal_gdl(gen_images, gt_images)
    loss = l2Loss + l1Loss + gdlLoss
    
    tf.summary.scalar('l2_Loss', l2Loss)
    tf.summary.scalar('l1_Loss', l1Loss)
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