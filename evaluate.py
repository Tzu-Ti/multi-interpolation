import os
import imageio
import tensorflow as tf

# files folder
file_folder_path = "results/test/kth_bi_lstm_t1s3_rgb/images0"

actions = []
for action in os.listdir(file_folder_path):
    actions.append(os.path.join(file_folder_path, action))

def evaluate(a, b):
    return (tf.image.psnr(a, b, max_val=255), tf.image.ssim(a, b, max_val=255))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    all_psnr = []
    all_ssim = []
    for folder in actions:
        print(folder)
        # all test file
        gt_names = ['gt%d.png' %(i+1) for i in range(0, 11)]
        pd_names = ['pd%d.png' %(i+1) for i in range(0, 11)]
        
        psnr_list = []
        ssim_list = []
        for gt_name, pd_name in zip(gt_names, pd_names):
            gt_path = os.path.join(folder, gt_name)
            pd_path = os.path.join(folder, pd_name)

            gt = tf.image.decode_image(tf.read_file(gt_path))
            pd = tf.image.decode_image(tf.read_file(pd_path))

            psnr, ssim = sess.run(evaluate(gt, pd))
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        print(psnr_list)
        all_psnr.append(psnr_list)
        all_ssim.append(ssim_list)
    
print("PSNR: ", all_psnr)
print("SSIM: ", all_ssim)

