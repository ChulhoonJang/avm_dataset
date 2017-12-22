# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:27:21 2017

@author: SRG
"""

import cv2
import os

OUTPUT_DIR = 'E:/ParkingSlotDetection/db/[20171222]_LG_AVM/rectified/set1'
if os.path.isdir(OUTPUT_DIR) == False:
    os.mkdir(OUTPUT_DIR)
    
cap = cv2.VideoCapture('E:/ParkingSlotDetection/db/[20171222]_LG_AVM/2017.12.22 09.38.45.mp4')

idx = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if frame is not None:
        frame[315:781,856:1067,:] = 0
        img = frame[:,663:1260,:]
        img = cv2.resize(cv2.flip(cv2.transpose(img),0), (392,240))
        cv2.imwrite('{}/{:08d}.png'.format(OUTPUT_DIR,idx), img)
        cv2.imshow('frame',img)
        idx += 1    
        cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows()