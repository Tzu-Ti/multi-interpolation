#%%
import os
import glob
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
#%%
def sortKeyFunc(s):
        return int(os.path.basename(s)[:-4])

def img_to_npz(path):
    img_file_count = 0
    num_of_files = 0
    for directory in ['NCTU_Ped_Oral_Demo_png']:
        set_path = os.path.join(path, directory)
        files = os.listdir(set_path)
        for file in files:
            num_of_files += len(glob.glob(set_path + '/' + file + '/*.png'))
    print(num_of_files)

    imgs = np.zeros(shape=(num_of_files,3,128,220))
    for directory in ['NCTU_Ped_Oral_Demo_png']:
        set_path = os.path.join(path, directory)
        files = os.listdir(set_path)
        for file in files:
            sortedglob = (glob.glob(set_path + '/' + file + '/*.png'))
            sortedglob.sort(key=sortKeyFunc)
            for img_file in tqdm(sortedglob):   ## 001.png to xxx.png
                img = cv2.imread(img_file)  #,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(220,128))
                img = np.transpose(img,(2,0,1))  # take RGB to 1st dim
                img = img / 255.0
                imgs[img_file_count] = img
                img_file_count += 1
    
    clips_arr = np.array([[[be*15+(io*15+ind*30)*abs(be-1) for be in range(2)] for ind in range(int(num_of_files/30))] for io in range(2)])
    np.savez('nctuPed_Oral_Demo_128x220_30f',clips=clips_arr,input_raw_data=imgs,dims=np.array([[3,128,220]]))

def main():
    for directory in ['']:
        project_path = './'
        image_path = os.path.join(project_path, directory)
        img_to_npz(image_path)
        
    print('Successfully converted img to npz.')


main()