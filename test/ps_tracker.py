# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:29:58 2017

@author: chulh
"""
import numpy as np
import cv2

def getMovement(dt_yaw, yawrate, dt_v, v):    
    delta_yaw = dt_yaw * yawrate / 180 * np.pi  # unit: rad
    delta_travel_dist = dt_v * v # unit: meter
    
    return delta_yaw, delta_travel_dist


def compare(pts_1, pts_2, thresh):    
    diff = (pts_1 - pts_2).reshape((4,1))
    
    dist_1 = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1])
    dist_2 = np.sqrt(diff[2]*diff[2] + diff[3]*diff[3])
        
    if dist_1 <= thresh and dist_2 <= thresh:
        return True
    else:
        return False
    
def predictPosition(pt, delta_yaw, delta_travel, center=(0,0)):   
    x = pt[0]
    y = pt[1]
    
    # Translation
    x = x + delta_travel
    
    # Rotation
    x = x - center[0]
    y = y - center[1]
    
    rot_x = np.cos(delta_yaw) * x - np.sin(delta_yaw) * y
    rot_y = np.sin(delta_yaw) * x + np.cos(delta_yaw) * y
    
    x = rot_x + center[0]
    y = rot_y + center[1]
    
    return  x, y

def findNearPoint(ref, pts):    
    diff = np.array(pts) - np.array(ref)    
    dist = np.sqrt(np.sum(np.square(diff), axis=1))    
    
    return np.argmin(dist), min(dist)

class ps_tracker:
    def __init__(self, ID, states, F, P, Q, R, H):
        self.ID = ID
        self.states = np.array(states).reshape([4,1])      
        self.F = F
        self.P = P
        self.Q = Q
        self.R = R
        self.H = H
        self.thresh = 10
        self.status = 0
        
        self.states_old = []
        self.states_motion_compensated = []
        self.measurement = []
        self.condition = 1
        self.occupied = False
        
    def run(self, dYaw, dDist, center, pts):
        self.status = 0
        
        self.states_old = np.copy(self.states)
        self.compensate_motion(dYaw, dDist, center)
        self.states_motion_compensated = np.copy(self.states)
        
        self.predict()
        flag, y = self.filter_measurement(pts)
        if flag != 0:
            self.correct(y)
            self.measurement = y # for debug
            self.status = 1
        
        if self.P[0,0] > 0.01:
            self.condition = 0
        
    def compensate_motion(self, dYaw, dDist, center = (0,0) ):        
        x, y = predictPosition((self.states[0], self.states[1]), dYaw, dDist, center)
        self.states[0] = x
        self.states[1] = y
        
        x, y = predictPosition((self.states[2], self.states[3]), dYaw, dDist, center)
        self.states[2] = x
        self.states[3] = y        
    
    def predict(self):
        self.states = np.dot(self.F, self.states)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    
    def correct(self, y):
        # correction        
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R))        
        self.states = self.states + np.dot(K, y - np.dot(self.H,self.states))
        self.P = self.P - np.dot(np.dot(K,self.H),self.P)
    
    def filter_measurement(self, pts):
        # fine the nearest pt within thresh        
        flag = [False, False]
        idx1, dist = findNearPoint(self.states[0:2].reshape((1,2)), pts)        
        if dist <= self.thresh:
            flag[0] = True
        
        idx2, dist = findNearPoint(self.states[2:4].reshape((1,2)), pts)
        if dist <= self.thresh:
            flag[1] = True
        
        if flag[0] == False and flag[1] == False:
            return 0, np.zeros((4,1))
        
        elif flag[0] == True and flag[1] == False:
            delta = np.array((self.states[0], self.states[1])).reshape(1,2) - pts[idx1]
            estimated = np.array((self.states[2], self.states[3])).reshape(1,2) - delta
            return 1, np.array((pts[idx1].reshape(1,2), estimated)).reshape(4,1)        
        
            delta = np.array([self.states[0], self.states[1]]).reshape(1,2) - pts[idx1].reshape(1,2)
            estimated = np.array([self.states[2], self.states[3]]).reshape(1,2) - delta 
            return 2, np.array((pts[idx1].reshape(1,2), estimated)).reshape(4,1)
            
        elif flag[0] == False and flag[1] == True:
            delta = np.array([self.states[2], self.states[3]]).reshape(1,2) - pts[idx2].reshape(1,2)
            estimated = np.array([self.states[0], self.states[1]]).reshape(1,2) - delta 
            return 2, np.array((estimated, pts[idx2].reshape(1,2))).reshape(4,1)
        
        elif flag[0] == True and flag[1] == True:
            return 3, np.array((pts[idx1], pts[idx2])).reshape(4,1)
            
    def get_track(self):
        if self.P[0,0] < (2.8e-04):
            return True, self.states.reshape((2,2))
        else:
            return False, []
        
    def get_states(self):        
        return self.states.reshape((2,2))
    
    def plot(self, img, r):
        pts = self.states_old
        if len(pts) == 4:        
            img = cv2.circle(img, (np.int(pts[0]), np.int(pts[1])),r, (0,0,255), -1) # blue
            img = cv2.circle(img, (np.int(pts[2]), np.int(pts[3])),r, (0,0,255), -1) # blue
       
        pts = self.states_motion_compensated
        if len(pts) == 4: 
            img = cv2.circle(img, (np.int(pts[0]), np.int(pts[1])),r, (0,255,0), -1) # green
            img = cv2.circle(img, (np.int(pts[2]), np.int(pts[3])),r, (0,255,0), -1) # green
        
        pts = self.measurement
        if self.status == 1:            
            img = cv2.circle(img, (np.int(pts[0]), np.int(pts[1])),r, (255,255,0), -1) # yellow
            img = cv2.circle(img, (np.int(pts[2]), np.int(pts[3])),r, (255,255,0), -1) # yellow
        
        pts = self.states
        if len(pts) == 4: 
            img = cv2.circle(img, (np.int(pts[0]), np.int(pts[1])),r, (255,0,0), -1)
            img = cv2.circle(img, (np.int(pts[2]), np.int(pts[3])),r, (255,0,0), -1)
            
        return img

