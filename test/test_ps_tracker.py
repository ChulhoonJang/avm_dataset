# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 19:07:54 2017

@author: chulh
"""

import os
import pickle
from skimage.io import imread
import matplotlib.pyplot as plt
from radon_transform import drawRectangle, space_estimation
import cv2
import numpy as np
from ps_tracker import compare, ps_tracker, getMovement

set_num = 3
img_path = 'C:/Users/chulh/문서/Git/avm_dataset/dataset/hyu_171121/rectified/set{}'.format(set_num)
data_path = 'C:/Users/chulh/문서/Git/avm_dataset/dataset/hyu_171121/motion/set{}'.format(set_num)
meas_path = 'C:/Users/chulh/Documents/hyu_171121/set{}'.format(set_num)

CalibPara = pickle.load(open(os.path.join(data_path, 'calib.p'),"rb"))
Motion = pickle.load(open(os.path.join(data_path, 'motion.p'),"rb"))
Measurments = pickle.load(open(os.path.join(meas_path, 'measurement.p'),"rb"))

MeterPerPixel = CalibPara['meter_per_pixel']
XPixelMeter = MeterPerPixel[0][0]
YPixelMeter = MeterPerPixel[0][1]
VehicleCenter = CalibPara['center']
VehicleCenterRow = VehicleCenter[0][1]
VehicleCenterCol = VehicleCenter[0][0]
ItoW = CalibPara['ItoW']
WtoI = CalibPara['WtoI']

SpeedData = Motion['speed']
YawRateData = Motion['yaw_rate']

frames = [602, 442, 539, 454, 557, 546, 581, 541]
f0 = 0
f1 = frames[set_num-1]

tracks = []
thresh = 10

F = 1. * np.eye((4))
H = 1. * np.eye((4))
Q = 0.5e-4 * np.eye((4))  # process noise covariance
R = 1e-3 * np.eye((4))  # observation noise covariance
P = 2e-1 * np.eye((4))     # posteriori error covariance
    
for i in range(f0,f1):
    
    img = imread(os.path.join(img_path, '{:08d}.jpg'.format(i)))
    measurement = Measurments[i]
    dYaw, dDist = getMovement(YawRateData[i][0] - YawRateData[i-1][0], YawRateData[i][1],
                              SpeedData[i][0] - SpeedData[i-1][0], SpeedData[i][1])
    
    dDist_pixel = dDist / XPixelMeter
    # tracking
    new_ps = []
    for ps in measurement['pss']:
        bNewMeas = True
        for track in tracks:
            if compare(ps[0:2,:], track.get_states(), thresh) == True:
                bNewMeas = False       
        if bNewMeas == True:
            new_ps.append(ps)
    
    # tracking
    for track in tracks:
        track.run(dYaw, dDist_pixel,(VehicleCenterCol, VehicleCenterRow), measurement['pts'])
        
    # new track initilization
    for ps in new_ps:
        init_states = ps[0:2,:]
        tracks.append(ps_tracker(init_states, F, P, Q, R, H))
        
   
    # draw    
#    img_debug = img
#    for ps in new_ps:
#        img_debug = drawRectangle(img_debug, ps, 3)
#    plt.imshow(img)
#    plt.show()
    
#    img_debug = img
#    for ps in measurement['pss']:
#        img_debug = drawRectangle(img_debug, ps, 3)
#    for pt in measurement['pts']:
#         img_debug = cv2.circle(img_debug, (np.int(pt[0]), np.int(pt[1])),3, (0,255,255), -1)
#         
#    plt.imshow(img)
#    plt.show()
    img_debug = np.copy(img)
    for track in tracks:
        flag, pts = track.get_track()
        if flag == True:
            ps = space_estimation(pts[0,:], pts[1,:], 5. / XPixelMeter)
            img = drawRectangle(img_debug, ps, 3)
            
        if track.condition == 0:
            tracks.remove(track)
            print('track is removed')
            #print(track.P)
#        img_debug = np.copy(img)
#        img_debug = track.plot(img_debug,3)        
    
    cv2.imshow('img_debug',img_debug)
    cv2.waitKey(33)
   

cv2.destroyAllWindows()
#
#i = 0
#for track in tracks:
#    i = i + 1
#    print(i)
#    print(track.P)
#
#img_debug = np.copy(img)
#img_debug = tracks[6].plot(img_debug, 3)
#plt.imshow(img_debug)
#plt.show()