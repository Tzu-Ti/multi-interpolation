import os
import imageio
import tensorflow as tf

# all iteration path
iteration_path = 'results/kth_bi_lstm'
iteration_folder = os.listdir(iteration_path)

# remove .ipynb_checkpoints
iteration_folder = iteration_folder[1:]

iteration_folder.sort(key=lambda s: int(s))
last_iteration = iteration_folder[-1]
print(last_iteration)

# last iteration path
last_iteration_path = os.path.join(iteration_path, last_iteration)

# last iteration folder
last_iteration_folder = os.listdir(last_iteration_path)
last_iteration_folder.sort()

# remove .ipynb_checkpoints
last_iteration_folder = last_iteration_folder[1:]

# first test result
file_folder_name = last_iteration_folder[0]
file_folder_path = os.path.join(last_iteration_path, file_folder_name)

# all test file
gt_names = ['gt%d.png' %(i+1) for i in range(0, 11)]
pd_names = ['pd%d.png' %(i+1) for i in range(0, 11)]

gt = []
pd = []
for filename in gt_names:
    gt_path = os.path.join(file_folder_path, filename)
    gt.append(imageio.imread(gt_path))
imageio.mimsave('gt.gif', gt, duration = 0.5)

for filename in pd_names:
    pd_path = os.path.join(file_folder_path, filename)
    pd.append(imageio.imread(pd_path))
imageio.mimsave('pd.gif', pd, duration = 0.5)

def evaluate(a, b):
    return (tf.image.psnr(a, b, max_val=255), tf.image.ssim(a, b, max_val=255))

psnr_list = []
ssim_list = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for gt_name, pd_name in zip(gt_names, pd_names):
        gt_path = os.path.join(file_folder_path, gt_name)
        pd_path = os.path.join(file_folder_path, pd_name)

        gt = tf.image.decode_image(tf.read_file(gt_path))
        pd = tf.image.decode_image(tf.read_file(pd_path))
    
        psnr, ssim = sess.run(evaluate(gt, pd))
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    
print("PSNR: ", psnr_list)
print("SSIM: ", ssim_list)
