# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:13:38 2017

@author: chulh
"""
import peakutils
import numpy as np
import matplotlib.pyplot as plt
import cv2

def findCrossPoint(L1, L2):
    x = (L1[1] - L2[1]) / (L2[0] - L1[0])
    y = -(L1[0] * L2[1] - L2[0] * L1[1]) / (L2[0] - L1[0])    
    return (x, y)

def generateRadonLines(radon_prj, theta, ang_idx, thresh, dist, center, img_size, debug = False ):
    prj = radon_prj[:, ang_idx].reshape(radon_prj.shape[0],)
    indexes = peakutils.indexes(prj, thres=thresh, min_dist=dist)
    
    radon_lines = []
    for idx in indexes:
        l = ps_line([(20, 30), (48, 60)])
        l.generate(idx, theta[ang_idx], center, img_size)
        radon_lines.append(l)
    
    if debug == True:
        plt.plot(prj)        
        for idx in indexes:
            plt.plot(idx, prj[idx],'.')
        plt.show()
    
    return radon_lines

def createTemplate(src, dist, margin):
    img_lower = np.zeros(src.shape, np.uint8)
    img_lower = cv2.line(img_lower, (0, margin), (margin*2+np.int(dist), margin), (255,255,255), 2)
    img_lower = cv2.line(img_lower, (margin, margin), (margin, margin*2), (255,255,255), 2)
    img_lower = cv2.line(img_lower, (margin+np.int(dist), margin), (margin+np.int(dist), margin*2), (255,255,255), 2)

    img_upper = np.zeros(src.shape, np.uint8)
    img_upper = cv2.line(img_upper, (0, margin), (margin*2+np.int(dist), margin), (255,255,255), 2)
    img_upper = cv2.line(img_upper, (margin, margin), (margin, 0), (255,255,255), 2)
    img_upper = cv2.line(img_upper, (margin+np.int(dist), margin), (margin+np.int(dist), 0), (255,255,255), 2)

    img_template = [img_lower, img_upper]    
    
    return img_template


def drawRectangle(src, pts):
    dst = src
    
    dst = cv2.line(dst, 
                  (np.int(pts[0][0]), np.int(pts[0][1])),
                  (np.int(pts[1][0]), np.int(pts[1][1])),
                  (255,0,0), 1)
    dst = cv2.line(dst, 
                  (np.int(pts[1][0]), np.int(pts[1][1])),
                  (np.int(pts[2][0]), np.int(pts[2][1])),
                  (255,0,0), 1)
    dst = cv2.line(dst, 
                  (np.int(pts[2][0]), np.int(pts[2][1])),
                  (np.int(pts[3][0]), np.int(pts[3][1])),
                  (255,0,0), 1)
    dst = cv2.line(dst, 
                  (np.int(pts[3][0]), np.int(pts[3][1])),
                  (np.int(pts[0][0]), np.int(pts[0][1])),
                  (255,0,0), 1)
    return dst

def calculateMarginalSpace(pt1, pt2, margin):            
    ang = 90 / 180 * np.pi # degree > radian
    
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
       
    dist = np.sqrt(dx*dx + dy * dy)
       
    # unit vector
    ux = dx/dist
    uy = dy/dist
    
    # rotation
    ux_rot = np.cos(ang) * ux - np.sin(ang) * uy
    uy_rot = np.sin(ang) * ux + np.cos(ang) * uy
    
    pt1 = (pt1[0] - margin * ux, pt1[1] - margin * uy)
    pt2 = (pt2[0] + margin * ux, pt2[1] + margin * uy)
    
    point1 = (pt1[0] - margin * ux_rot, pt1[1] - margin * uy_rot)
    point2 = (pt2[0] - margin * ux_rot, pt2[1] - margin * uy_rot)
    point3 = (pt2[0] + margin * ux_rot, pt2[1] + margin * uy_rot)
    point4 = (pt1[0] + margin * ux_rot, pt1[1] + margin * uy_rot)
    
    space = [point1, point2, point3, point4]    
    
    return space, dist
    
def cropImage(space, img, img_crop_size):    
    arr_space = np.array(space)
        
    point1 = (arr_space[0,0], arr_space[0,1])
    point2 = (arr_space[1,0], arr_space[1,1])
    point3 = (arr_space[2,0], arr_space[2,1])
    point4 = (arr_space[3,0], arr_space[3,1])
    
    CropXLen = np.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)
    CropYLen = np.sqrt((point3[0]-point2[0])**2 + (point3[1]-point2[1])**2)
    
    pt_origin = [point1[0], point1[1], point2[0], point2[1], point3[0], point3[1], point4[0], point4[1]]
    pt_origin = np.array(pt_origin, dtype=np.float32).reshape((4,2))
    
    pt_transform = [0, 0, CropXLen, 0, CropXLen, CropYLen, 0, CropYLen]
    pt_transform = np.array(pt_transform, dtype=np.float32).reshape((4,2))
    
    Trans_mat = cv2.getPerspectiveTransform(pt_origin,pt_transform)
    img_crop = cv2.warpPerspective(img, Trans_mat, img_crop_size)
    
    return img_crop
    
class radon_line:
    def __init__(self):
        self.ang = 0.
        self.pos = [0.0, 0.0]
        self.pts = []
        
    def generate(self, offset, ang_ps1, center, img_size):
        # angle
        if ang_ps1 <= 90:
            self.ang = 90 - ang_ps1
        else:
            self.ang = 90 + (180 - ang_ps1)
        # point
        ang_oth_rad = (180. - ang_ps1) / 180. * np.pi
        self.pos[0] = img_size[0] // 2 - np.cos(ang_oth_rad) * (offset - center)        
        self.pos[1] = img_size[1] // 2 - np.sin(ang_oth_rad) * (offset -center)
        
        if (self.ang >= 0 and self.ang < 45 ) or (self.ang >= 135 and self.ang <= 180 ):
            # y = ax + b
            self.a = np.tan(self.ang / 180 * np.pi)
            self.b = self.pos[1] - self.pos[0] * self.a
            
            self.end_pts = []
            self.end_pts.append((0, self.a * 0 + self.b))
            self.end_pts.append((img_size[0], self.a * img_size[0] + self.b))
        else:
            # x = ay + b
            self.a = np.cos(self.ang / 180 * np.pi + 0.00000001) / np.sin(self.ang / 180 * np.pi + 0.00000001)
            self.b = self.pos[0] - self.pos[1] * self.a
            
            self.end_pts = []
            self.end_pts.append((self.a * 0 + self.b, 0))
            self.end_pts.append((self.a * img_size[1] + self.b, img_size[1]))
    def get_a(self):
        if (self.ang >= 0 and self.ang < 45 ) or (self.ang >= 135 and self.ang <= 180 ):
            return self.a
        else:
            return (1. / self.a)
            
    def get_b(self):
        if (self.ang >= 0 and self.ang < 45 ) or (self.ang >= 135 and self.ang <= 180 ):
            return self.b
        else:
            return (- self.b / self.a)
    
        
class ps_line(radon_line):
    def __init__(self, length_limit):
        radon_line.__init__(self)
        self.pts = []
        self.valid_pts = []
        self.invalid_pts = []
        self.per_pairs = [] # perpendicular
        self.par_pairs = [] # parallel
        self.length_limit = {'perpendicular': length_limit[0], 'parallel': length_limit[1]}
        self.per_spaces = []
        
        
    def set_cross_pt(self, pt):
        self.pts.append(pt)
    
    def classify_valid_cross_pt(self, img):
        for pt in self.pts:
            if img[np.int(pt[1]), np.int(pt[0])] == 0:
                self.invalid_pts.append(pt)
            else:
                self.valid_pts.append(pt)
        self.valid_pts = sorted(self.valid_pts, key=lambda tup: tup[0])               
        self.invalid_pts = sorted(self.invalid_pts, key=lambda tup: tup[0])
    
    def generate_pairs(self):
        if len(self.pts) >= 2:
            for i in range(len(self.pts)-1):
                for j in range(i,len(self.pts)):
                    p1 = self.pts[i]
                    p2 = self.pts[j]
                    dist = np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))
                    
                    if dist >= self.length_limit['perpendicular'][0] and dist <= self.length_limit['perpendicular'][1]:
                        pairs = (p1, p2)
                        self.per_pairs.append(pairs)
                        #print('perpendicular: {:2.2f} pixels'.format(dist))
                    elif dist >= self.length_limit['parallel'][0] and dist <= self.length_limit['parallel'][1]:
                        pairs = (p1, p2)
                        self.par_pairs.append(pairs)
                       # print('parallel: {:2.2f} pixels'.format(dist))
                       
    def classify_pairs(self, img, margin, debug = False):
        for pair in self.per_pairs:
            marginal_space, dist = calculateMarginalSpace(pair[0], pair[1], margin)
            img_crop = cropImage(marginal_space, img, (np.int(dist+margin*2), margin*2))
            slot_temp = createTemplate(img_crop, dist, margin)
            slot_temp = createTemplate(img_crop, dist, margin)            
            
            results = []
            for template in slot_temp:
                result = cv2.matchTemplate(img_crop, template, cv2.TM_CCORR_NORMED)
                results.append(result)
            
            
            
            if debug == True:
                plt.imshow(img_crop, cmap = 'gray')                
                plt.show()
                for template, result in zip(slot_temp, results):    
                    plt.imshow(template, cmap = 'gray')    
                    plt.show()              
                    print(result)
            
        for pair in self.par_pairs:
            marginal_space, dist = calculateMarginalSpace(pair[0], pair[1], margin)
            img_crop = cropImage(marginal_space, img, (np.int(dist+margin*2), margin*2))
            slot_temp = createTemplate(img_crop, dist, margin)
            slot_temp = createTemplate(img_crop, dist, margin)            
            
            results = []
            for template in slot_temp:
                result = cv2.matchTemplate(img_crop, template, cv2.TM_CCORR_NORMED)
                results.append(result)
            
            if debug == True:
                plt.imshow(img_crop, cmap = 'gray')                
                plt.show()
                for template, result in zip(slot_temp, results):    
                    plt.imshow(template, cmap = 'gray')    
                    plt.show()              
                    print(result)
        