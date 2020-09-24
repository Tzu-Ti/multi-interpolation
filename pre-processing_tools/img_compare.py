#%%
import os
import glob
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from functools import reduce
from PIL import Image
#%%

def phash(img):
    """
    :param img: image
    :return: image local hash value
    """
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    hash_value=reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0)
    #print(hash_value)
    return hash_value


def hamming_distance(a, b):
    """
    :param a: image 1 hash
    :param b: image 2 hash
    :return: two images hamming distance of hash value
    """
    hm_distance=bin(a ^ b).count('1')
    #print(hm_distance)
    return hm_distance


def is_imgs_similar(img1,img2):
    """
    :param img1: image1
    :param img2: image2
    :return:  True: similar  False: not similar
    """
    return True if hamming_distance(phash(img1),phash(img2)) <= 2 else False


def img_compare(path):
    files = os.listdir(path)
    sensitive_pic = []

    for file in tqdm(files):
        sensi_files = os.listdir(os.path.join('./sensitive_pics/', file))
        for sensi_file in sensi_files:
            sensitive_pic.append(Image.open(os.path.join('./sensitive_pics/', file, sensi_file)))
        del_cnt = 0
        if file in ['666']:
            for img_file in tqdm(sorted(glob.glob(path + '/' + file + '/*.jpg'))):   ## 001.jpg to xxx.jpg
                theSame = []
                target_pic = Image.open(img_file)
                for i_pic in range(0,10,2):
                    theSame.append(is_imgs_similar(target_pic, sensitive_pic[i_pic])) # = is_imgs_similar(target_pic, sensitive_pic1)
                if True in theSame:
                    del_cnt += 1
                    os.remove(img_file)
                #print(img_file, theSame)
            print(del_cnt)


def main():
    for directory in ['dupeguru3']:
        project_path = './'
        image_path = os.path.join(project_path, directory)
        img_compare(image_path)
        
    print('Successfully deleted similar images.')


main()