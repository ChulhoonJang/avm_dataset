# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 18:30:19 2018

@author: chulh
"""
#from skimage.measure import compare_ssim
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile

IMG_DIR = 'C:/Users/chulh/Downloads/rectified/set7'
OUTPUT_DIR = 'C:/Users/chulh/Downloads/rectified/selected'
files = os.listdir(IMG_DIR)

f = 0
img_ref = cv2.imread(os.path.join(IMG_DIR, files[f]))
img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

frames = []
frames.append(files[f])
for i in range(f,len(files)):
    
    file = files[i]
    
    if len(file.split('.')) != 2 or file.split('.')[1] != 'jpg':
        continue
    
    img = cv2.imread(os.path.join(IMG_DIR, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    score = np.sum(np.abs(img_ref.astype(np.double) - img.astype(np.double)))/(img.shape[0]*img.shape[1])
        
    if score > 20:
        frames.append(file)
        print('frame: {}'.format(i+f))
        plt.imshow(img, cmap='gray')        
        plt.show()
        img_ref = img
        
idx = 135
for frame in frames:
    copyfile(os.path.join(IMG_DIR, frame), os.path.join(OUTPUT_DIR,'{:08d}.png'.format(idx)))
    idx += 1
    
print(idx)