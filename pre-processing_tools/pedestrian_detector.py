#%%
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import os
import glob
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
#%%


def ped_detect(path):
    files = os.listdir(path)
    sensitive_pic = []

    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--images", required=True, help="path to images directory")
    # args = vars(ap.parse_args())

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    for file in tqdm(files):
        del_cnt = 0
        if file in ['666']:
            #for img_file in tqdm(sorted(glob.glob(path + '/' + file + '/*.jpg'))):   ## 001.jpg to xxx.jpg
            
            # loop over the image paths
            for imagePath in paths.list_images(path + '/' + file):
                # load the image and resize it to (1) reduce detection time
                # and (2) improve detection accuracy
                image = cv2.imread(imagePath)
                #image = imutils.resize(image, width=min(400, image.shape[1]), height=min(200, image.shape[0]))
                orig = image.copy()

                #rects_cnt = 0
                # detect people in the image
                (rects, weights) = hog.detectMultiScale(image, winStride=(3, 3),
                    padding=(16, 16), scale=1.05)

                # draw the original bounding boxes
                for (x, y, w, h) in rects:
                    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    #rects_cnt += 1

                # apply non-maxima suppression to the bounding boxes using a
                # fairly large overlap threshold to try to maintain overlapping
                # boxes that are still people
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

                # draw the final bounding boxes
                for (xA, yA, xB, yB) in pick:
                    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

                # show some information on the number of bounding boxes
                filename = imagePath[imagePath.rfind("/") + 1:]
                print("[INFO] {}: {} original boxes, {} after suppression".format(
                    filename, len(rects), len(pick)))

                # show the output images
                # cv2.imshow("Before NMS", orig)
                # cv2.imshow("After NMS", image)
                # cv2.waitKey(0)


                if len(pick) > 0 :
                    del_cnt += 1
                    #os.remove(imagePath)
                    #cv2.imshow("Before NMS", orig)
                    cv2.imwrite(imagePath, image)

                #print(imagePath, theSame)
            print(del_cnt)

        


def main():


    for directory in ['']:
        project_path = './'
        image_path = os.path.join(project_path, directory)
        ped_detect(image_path)
        
    print('Successfully deleted similar images.')


main()