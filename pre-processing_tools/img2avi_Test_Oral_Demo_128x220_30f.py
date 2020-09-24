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

def img_to_avi(path):
    img_file_count = 0
    num_of_files = 0
    for directory in ['NCTU_Ped_Oral_Demo_png_Test_TAIdata']:
        set_path = os.path.join(path, directory)
        files = os.listdir(set_path)
        for file in files:
            num_of_files += len(glob.glob(set_path + '/' + file + '/*.png'))
    print(num_of_files)

    imgs = np.zeros(shape=(num_of_files,128,220,3))
    for directory in ['NCTU_Ped_Oral_Demo_png_Test_TAIdata']:
        set_path = os.path.join(path, directory)
        files = os.listdir(set_path)
        for file in files:
            sortedglob = (glob.glob(set_path + '/' + file + '/*.png'))
            sortedglob.sort(key=sortKeyFunc)
            for img_file in tqdm(sortedglob):   ## 001.png to xxx.png
                img = cv2.imread(img_file)  #,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(220,128))
                #img = np.transpose(img,(2,0,1))  # take RGB to 1st dim
                #img = img / 255.0
                imgs[img_file_count] = img
                img_file_count += 1
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('nctuPed_Test_Oral_Demo_128x220.avi', fourcc, 10.0, (220,128))
    for img in imgs:
        out.write(img.astype(np.uint8))
    out.release()

def main():
    for directory in ['']:
        project_path = './'
        image_path = os.path.join(project_path, directory)
        img_to_avi(image_path)
        
    print('Successfully converted img to avi.')


main()