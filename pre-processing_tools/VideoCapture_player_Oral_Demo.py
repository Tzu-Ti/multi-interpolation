#%%
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import os
import shutil
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

#%%
def ped_hand_detect(video_images):
    # initialize the HOG descriptor/person detector
    ped_flag = False
    clip_start_end = [[] for i in range(2)]
    curr = 0
    win_name = 'Jaden Player'
    while(True):
        if curr > len(video_images)-1:
            curr = len(video_images)-1
        if curr < 0:
            curr = 0
        #curr = curr % len(video_images)
        image = video_images[curr]
        # print(i_image)
        cv2.imshow(win_name, image)
        cv2.waitKey(1)
        #wKEx = cv2.waitKeyEx(1)
        print( round(curr/len(video_images)*100), '% (', curr, '/', len(video_images), ')')
        wK = cv2.waitKeyEx(-1)
        if wK == ord('d') or wK == 2555904:
            curr += 7
            continue
        elif wK == ord('a') or wK == 2424832:
            curr -= 7
            continue
        elif wK == ord('c'):
            curr += 3
            continue
        elif wK == ord('z'):
            curr -= 3
            continue
        elif wK == ord('x'):
            print(clip_start_end[0])
            print(clip_start_end[1])
            continue
        elif wK == ord('r'):
            if len(clip_start_end[0]) > len(clip_start_end[1]):
                if clip_start_end[0]:
                    clip_start_end[0].pop(-1)
                    ped_flag = False
            else:
                if clip_start_end[1]:
                    clip_start_end[1].pop(-1)
                    ped_flag = True
            print(clip_start_end[0])
            print(clip_start_end[1])
            continue
        elif wK == ord(' '):
            if ped_flag == False:
                ped_flag = True
                clip_start_end[0].append(curr)
                print(clip_start_end[0])
                print(clip_start_end[1])
                continue
            if ped_flag == True:
                ped_flag = False
                clip_start_end[1].append(curr)
                print(clip_start_end[0])
                print(clip_start_end[1])
                continue
        elif wK == ord('w'):
            if ped_flag == False:
                ped_flag = True
                clip_start_end[0].append(curr)
                print(clip_start_end[0])
                print(clip_start_end[1])
            continue
        elif wK == ord('s'):
            if ped_flag == True:
                ped_flag = False
                clip_start_end[1].append(curr)
                print(clip_start_end[0])
                print(clip_start_end[1])
            continue
        elif (wK == 27 or wK == ord('q')):
            break
    cv2.destroyAllWindows()

    return clip_start_end

#%%
def writeImg(video_images, clip_start_end, video_name):
    frame_index = []
    mid_clip_range = 0
    clip_ranges = []
    shape_clip_s_e = np.array(clip_start_end)
    shape_clip_s_e = shape_clip_s_e.shape[-1]
    for i_clip in range(shape_clip_s_e):
        clip_ranges.append(clip_start_end[1][i_clip] - clip_start_end[0][i_clip] + 1)
    clip_ranges.sort()
    num_of_clips = len(clip_ranges)
    mid_clip_range = clip_ranges[int(len(clip_ranges)/2)]
    for i_clip in range(shape_clip_s_e):
        clip_range = clip_start_end[1][i_clip] - clip_start_end[0][i_clip] + 1
        print('clip_range = ',clip_range)
        
        if clip_range < mid_clip_range*0.6 and num_of_clips > 6:
            num_of_clips -= 1
            print('removed')
            continue
        if clip_range < 30:
            frame_add = round((30 - (clip_range + 0.1))/2)
            clip_start_end[1][i_clip] += frame_add
            clip_start_end[0][i_clip] -= frame_add
            clip_range = clip_start_end[1][i_clip] - clip_start_end[0][i_clip] + 1

        for i_fp30f in range(30):
            i_fp30f = i_fp30f*clip_range/30
            i_fp30f = round(i_fp30f)
            frame_index.append(int(clip_start_end[0][i_clip] + i_fp30f))
    video_name_sp = video_name.split('\\')          # for Windows
    video_name_sp2 = video_name_sp[1].split('.')  # for Windows
    # video_name_sp = video_name.split('/')         # for Linux
    # video_name_sp2 = video_name_sp[2].split('.')  # for Linux
    video_name = video_name_sp2[0]
    if os.path.exists('NCTU_Ped_Oral_Demo_png/'+ video_name):
        shutil.rmtree('NCTU_Ped_Oral_Demo_png/'+ video_name)
    os.makedirs('NCTU_Ped_Oral_Demo_png/'+ video_name, mode=0o777)
    for i_write in frame_index:
        selected_frame = video_images[i_write]
        selected_frame = cv2.resize(selected_frame,(228,128))
        cv2.imwrite('NCTU_Ped_Oral_Demo_png/'+ video_name +'/'+ str(i_write)+'.png', selected_frame)

#%%


#%%
def get_images_from_video(video_name, time_F):
    video_images = []
    vc = cv2.VideoCapture(video_name)
    # vc.set(cv2.CAP_PROP_FRAME_WIDTH, 228)
    # vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)
    c = 1
    
    if vc.isOpened(): 
        rval, video_frame = vc.read()
    else:
        rval = False

    while rval:   
        rval, video_frame = vc.read()
        #time.sleep(0.01)
        #print(c)
        if(rval and c % time_F == 0): 
            video_frame = cv2.resize(video_frame,(435,256))
            video_frame = cv2.flip(video_frame, -1)
            video_images.append(video_frame)     
        c = c + 1
    vc.release()
    
    return video_images

#%% Main
time_F = 2              
video_root_path = './NCTU_Ped_Oral_Demo_MOV'
#folders = os.listdir(video_root_path)
video_file_count = 0
num_of_files = len(glob.glob(video_root_path + '/*.MOV'))
print('num of files =', num_of_files)
print('''
    go forward: '→', 'd', 'c'(slower)
    go backward: '←', 'a', 'z'(slower)
    set start/end flag: 'space'
    set start flag: 'w'
    set end flag: 's'
    cancel last flag: 'r'
    display current clips: 'x'
    quit: 'q'
    ''')
for video in tqdm(sorted(glob.glob(video_root_path + '/*.MOV'))):
    #if 'IMG_6154' in video:
    video_name = video
    print('video Name = ' + video_name)
    t_start = time.time()
    video_images = get_images_from_video(video_name, time_F) 
    t_vcend=time.time()
    print('vc Time = '+ str(t_vcend-t_start))
    clip_s_e = ped_hand_detect(video_images)
    t_pdend=time.time()
    print('pd Time = '+ str(t_pdend-t_vcend))
    writeImg(video_images, clip_s_e, video_name)
    t_wiend=time.time()
    print('wi Time = '+ str(t_wiend-t_pdend))
    video_file_count += 1
        
print('Successfully converted {video_file_count} videos.'.format(video_file_count=video_file_count))

# %%
