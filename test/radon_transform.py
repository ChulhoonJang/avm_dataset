# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:13:38 2017

@author: chulh
"""
import peakutils
import numpy as np

def findCrossPoint(L1, L2):
    x = (L1[1] - L2[1]) / (L2[0] - L1[0])
    y = -(L1[0] * L2[1] - L2[0] * L1[1]) / (L2[0] - L1[0])    
    return (x, y)

def generateRadonLines(radon_prj, ang, thresh, dist, center, img_size):
    prj = radon_prj[:, ang].reshape(radon_prj.shape[0],)
    indexes = peakutils.indexes(prj, thres=thresh, min_dist=dist)

    radon_lines = []
    for idx in indexes:
        l = radon_line()
        l.generate(idx, ang, center, img_size)
        radon_lines.append(l)
        
    return radon_lines
    
class radon_line:
    def __init__(self):
        self.ang = 0.
        self.pos = [0.0, 0.0]
        
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