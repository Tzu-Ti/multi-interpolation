__author__ = 'jaden'
#%%
import os.path
import time
import numpy as np
import tensorflow as tf
import cv2
import sys
import random
from nets import models_factory_bi_jstlstm
from data_provider import datasets_factory
from utils import preprocess
from utils import metrics
from skimage.measure import compare_ssim
#%%
# -----------------------------------------------------------------------------
FLAGS = tf.app.flags.FLAGS

# data I/O
tf.app.flags.DEFINE_string('dataset_name', 'BiTAI_base_dataset',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           'videolist/KTH/train_data_list.txt',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                        #    'videolist/KTH/random_files.txt',
                        #    'videolist/KTH/test_data_list_T=10.txt',
                           'videolist/KTH/random_092.txt',
                           'validation data paths.')
tf.app.flags.DEFINE_string('save_dir', '',
                            'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_frm_dir', 'results/kth_trape_bi_jstlstm_64b4_200K_infr092',
                           'dir to store result.')
tf.app.flags.DEFINE_string('log_dir', 'log/kth_trape_bi_jstlstm_64b4_200K_infr092', 
                            'log dir for TensorBoard')

# model
tf.app.flags.DEFINE_string('model_name', 'trape_bi_jstlstm',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', 'checkpoints/kth_trape_bi_jstlstm_64b4_200K/model.ckpt-200000',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 5,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 20,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('img_height', 128,
                            'input image width.')
tf.app.flags.DEFINE_integer('img_width', 128,
                            'input image height.')
tf.app.flags.DEFINE_integer('img_channel', 3,
                            'number of image channel.')
tf.app.flags.DEFINE_integer('stride', 1,
                            'stride of a convlstm layer.')
tf.app.flags.DEFINE_integer('filter_size', 5,
                            'filter of a convlstm layer.')
tf.app.flags.DEFINE_string('num_hidden', '64,64,64,64',
                           'COMMA separated number of units in a convlstm layer.')
tf.app.flags.DEFINE_integer('patch_size', 4,
                            'patch size on one dimension.')
tf.app.flags.DEFINE_boolean('layer_norm', True,
                            'whether to apply tensor layer norm.')
# optimization
tf.app.flags.DEFINE_float('lr', 0.0001,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')
tf.app.flags.DEFINE_integer('batch_size', 1,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 200000,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 1000,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 10000,
                            'number of iters saving models.')

class Model(object):
    def __init__(self):
        # inputs
        self.x = tf.placeholder(tf.float32,
                                [FLAGS.batch_size,
                                 FLAGS.seq_length,
                                 FLAGS.img_height/FLAGS.patch_size,
                                 FLAGS.img_width/FLAGS.patch_size,
                                 FLAGS.patch_size*FLAGS.patch_size*FLAGS.img_channel])

        self.x_rev = tf.placeholder(tf.float32,
                                [FLAGS.batch_size,
                                 FLAGS.seq_length,
                                 FLAGS.img_height/FLAGS.patch_size,
                                 FLAGS.img_width/FLAGS.patch_size,
                                 FLAGS.patch_size*FLAGS.patch_size*FLAGS.img_channel])

        self.mask_true = tf.placeholder(tf.float32,
                                        [FLAGS.batch_size,
                                         FLAGS.seq_length,
                                         FLAGS.img_height/FLAGS.patch_size,
                                         FLAGS.img_width/FLAGS.patch_size,
                                         FLAGS.patch_size*FLAGS.patch_size*FLAGS.img_channel])

        grads = []
        loss_train = []
        self.pred_seq = []
        self.pred_watch_ims_seq = []
        self.pred_watch_ims_rev_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        num_hidden = [int(x) for x in FLAGS.num_hidden.split(',')]
        print(num_hidden)
        num_layers = len(num_hidden)
        with tf.variable_scope(tf.get_variable_scope()):
            # define a model
            output_list = models_factory_bi_jstlstm.construct_model(
                FLAGS.model_name, self.x, self.x_rev,
                self.mask_true,
                num_layers, num_hidden,
                FLAGS.filter_size, FLAGS.stride,
                FLAGS.seq_length, FLAGS.input_length,
                FLAGS.layer_norm)
            gen_ims = output_list[0]
            loss = output_list[1]
            watch_ims = output_list[2]
            watch_ims_rev = output_list[3]
            pred_ims = gen_ims[:,FLAGS.input_length-4:(FLAGS.seq_length - FLAGS.input_length - 4)]
            pred_watch_ims = watch_ims[:,FLAGS.input_length-1:]
            pred_watch_ims_rev = watch_ims_rev[:,FLAGS.input_length-1:]
            self.loss_train = loss / FLAGS.batch_size
            # gradients
            all_params = tf.trainable_variables()
            grads.append(tf.gradients(loss, all_params))
            self.pred_seq.append(pred_ims)
            self.pred_watch_ims_seq.append(pred_watch_ims)
            self.pred_watch_ims_rev_seq.append(pred_watch_ims_rev)

        self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

         # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(visible_device_list='0')
        configProt = tf.ConfigProto(gpu_options=gpu_options)
        #configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.merged = tf.summary.merge_all()
        self.test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
        self.sess.run(init)
        if FLAGS.pretrained_model:
            self.saver.restore(self.sess, FLAGS.pretrained_model)
        
        # # # flops = tf.profiler.profile(self.sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        # # # params = tf.profiler.profile(self.sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        # # # print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

    def test(self, inputs, inputs_rev, mask_true, itr):
        feed_dict = {self.x: inputs, self.x_rev: inputs_rev}
        feed_dict.update({self.mask_true: mask_true})
        summary, gen_ims, watch_ims, watch_ims_rev = self.sess.run((self.merged, self.pred_seq, self.pred_watch_ims_seq, self.pred_watch_ims_rev_seq), feed_dict)
        self.test_writer.add_summary(summary, itr)
        return gen_ims, watch_ims, watch_ims_rev

def main(argv=None):
    if tf.gfile.Exists(FLAGS.gen_frm_dir):
        tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
    tf.gfile.MakeDirs(FLAGS.gen_frm_dir)
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        FLAGS.dataset_name, FLAGS.train_data_paths, FLAGS.valid_data_paths,
        FLAGS.batch_size, [FLAGS.img_height, FLAGS.img_width, FLAGS.img_channel], FLAGS.seq_length)

    print('Initializing models')
    model = Model()

#%%
    test_input_handle.begin(do_shuffle = False)
    totalDataLen = int(test_input_handle.total())
    print('totalDataLen=', totalDataLen)
    for itr in range(1):
        print('inference...')
        res_path = os.path.join(FLAGS.gen_frm_dir, 'images'+str(itr))
        os.mkdir(res_path)
        avg_mse = 0
        batch_id = 0
        img_mse,ssim,psnr,fmae,sharp= [],[],[],[],[]
        for i in range(FLAGS.seq_length - FLAGS.input_length -FLAGS.input_length):
            img_mse.append(0)
            ssim.append(0)
            psnr.append(0)
            fmae.append(0)
            sharp.append(0)

        mask_true = np.ones((FLAGS.batch_size,
                                FLAGS.seq_length,
                                FLAGS.img_height,
                                FLAGS.img_width,
                                FLAGS.img_channel))
        for num_batch in range(FLAGS.batch_size):
            for num_seq in range(FLAGS.seq_length):
                if num_seq < FLAGS.input_length or FLAGS.seq_length-1-num_seq < FLAGS.input_length:
                    continue
                else:
                    mask_true[num_batch,num_seq] = np.zeros((
                            FLAGS.img_height,
                            FLAGS.img_width,
                            FLAGS.img_channel))
                # for i in range(17,77):
                #     for j in range(57,87):
                #         mask_true[num_batch,num_seq,i,j,0] = 0
                #         mask_true[num_batch,num_seq,i,j,1] = 0
                #         mask_true[num_batch,num_seq,i,j,2] = 0
        mask_true = preprocess.reshape_patch(mask_true, FLAGS.patch_size)
        ###while(test_input_handle.no_batch_left() == False):
        while(batch_id < totalDataLen):
            batch_id = batch_id + 1
            print('test get_batch:')
            test_ims, fileName = test_input_handle.get_batch(False)
            test_dat = preprocess.reshape_patch(test_ims, FLAGS.patch_size)

            if FLAGS.reverse_input:
                test_ims_rev = test_dat[:,::-1]

            img_gen, ims_watch, ims_rev_watch = model.test(test_dat, test_ims_rev, mask_true, itr)

            # concat outputs of different gpus along batch
            img_gen = np.concatenate(img_gen)
            img_gen = preprocess.reshape_patch_back(img_gen, FLAGS.patch_size)
            ims_watch = np.concatenate(ims_watch)
            ims_watch = preprocess.reshape_patch_back(ims_watch, FLAGS.patch_size)
            ims_rev_watch = np.concatenate(ims_rev_watch)
            ims_rev_watch = preprocess.reshape_patch_back(ims_rev_watch, FLAGS.patch_size)
            # MSE per frame
            for i in range(FLAGS.seq_length - FLAGS.input_length-FLAGS.input_length):
                x = test_ims[:,i + FLAGS.input_length,:,:,0]
                gx = img_gen[:,i,:,:,0]
                fmae[i] += metrics.batch_mae_frame_float(gx, x)
                gx = np.maximum(gx, 0)
                gx = np.minimum(gx, 1)
                mse = np.square(x - gx).sum()
                img_mse[i] += mse
                avg_mse += mse

                real_frm = np.uint8(x * 255)
                pred_frm = np.uint8(gx * 255)
                psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
                for b in range(FLAGS.batch_size):
                    sharp[i] += np.max(
                        cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b],3)))
                    score, _ = compare_ssim(pred_frm[b],real_frm[b],full=True)
                    ssim[i] += score

            # save prediction examples
            if batch_id < totalDataLen:
                path = os.path.join(res_path, str(fileName))
                os.mkdir(path)
                for i in range(FLAGS.seq_length):
                    name = 'gt_middle_%04d.png' %(i)
                    file_name = os.path.join(path, name)
                    img_gt = np.uint8(test_ims[0,i,:,:,:] * 255)
                    cv2.imwrite(file_name, img_gt)
                    
                for i in range(FLAGS.seq_length-FLAGS.input_length-FLAGS.input_length):
                    name = 'pred_middle_%04d.png' %(i+FLAGS.input_length)
                    file_name = os.path.join(path, name)
                    img_pd = img_gen[0,i,:,:,:]
                    img_pd = np.maximum(img_pd, 0)
                    img_pd = np.minimum(img_pd, 1)
                    img_pd = np.uint8(img_pd * 255)
                    cv2.imwrite(file_name, img_pd)
            
            test_input_handle.next()
        avg_mse = avg_mse / (batch_id*FLAGS.batch_size)
        print('mse per seq: ' + str(avg_mse))
        for i in range(FLAGS.seq_length - FLAGS.input_length-FLAGS.input_length):
            print(img_mse[i] / (batch_id*FLAGS.batch_size))
        psnr = np.asarray(psnr, dtype=np.float32)/batch_id
        fmae = np.asarray(fmae, dtype=np.float32)/batch_id
        ssim = np.asarray(ssim, dtype=np.float32)/(FLAGS.batch_size*batch_id)
        sharp = np.asarray(sharp, dtype=np.float32)/(FLAGS.batch_size*batch_id)
        print('psnr per frame: ' + str(np.mean(psnr)))
        for i in range(FLAGS.seq_length - FLAGS.input_length-FLAGS.input_length):
            print(psnr[i])
        print('fmae per frame: ' + str(np.mean(fmae)))
        for i in range(FLAGS.seq_length - FLAGS.input_length-FLAGS.input_length):
            print(fmae[i])
        print('ssim per frame: ' + str(np.mean(ssim)))
        for i in range(FLAGS.seq_length - FLAGS.input_length-FLAGS.input_length):
            print(ssim[i])
        print('sharpness per frame: ' + str(np.mean(sharp)))
        for i in range(FLAGS.seq_length - FLAGS.input_length-FLAGS.input_length):
            print(sharp[i])

if __name__ == '__main__':
    tf.app.run()

