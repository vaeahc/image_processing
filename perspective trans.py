#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:38:33 2019

@author: vaeahc
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time


img = cv2.imread('IMG_0381.tif', 0)
s = time.time()

row, col = np.shape(img)
mat_trans = cv2.getPerspectiveTransform(np.float32([[100, 130], [380, 135], [50, 590], [495, 550]]), np.float32([[0, 0], [col, 0], [0, row-1], [col-1, row-1]]))
trg = np.zeros([row, col])
s = time.time()
for i in range(row):
    for j in range(col):
        temp_mat = np.dot(np.linalg.inv(mat_trans), np.array([j, i, 1]))
        temp_x = temp_mat[0] / temp_mat[2]
        temp_y = temp_mat[1] / temp_mat[2]
        if temp_x < 0 or temp_x > col -1 or temp_y < 0 or temp_y > row - 1:
            continue
        '''
        a = int((math.ceil(temp_x) - temp_x) * 2048)
        b = 2048 - a
        c = int((math.ceil(temp_y) - temp_y) * 2048)
        d = 2048 - c
        f_up = a * img[int(temp_y), int(temp_x)] + b * img[math.ceil(temp_y), int(temp_x)]
        f_down = a * img[int(temp_y), math.ceil(temp_x)] + b * img[math.ceil(temp_y), math.ceil(temp_x)]
        f_center = c * f_up + d * f_down
        trg[i, j] = f_center >> 22
        '''
        f_up = (math.ceil(temp_x) - temp_x) * img[int(temp_y), int(temp_x)] + (temp_x - int(temp_x)) * img[math.ceil(temp_y), int(temp_x)]
        f_down = (math.ceil(temp_x) - temp_x) * img[int(temp_y), math.ceil(temp_x)] + (temp_x - int(temp_x)) * img[math.ceil(temp_y), math.ceil(temp_x)]
        f_center = (math.ceil(temp_y) - temp_y) * f_up + (temp_y - int(temp_y)) * f_down
        trg[i, j] = f_center
        
        #trg[i, j] = img[int(temp_y), int(temp_x)]


print(time.time() - s)               
plt.imshow(trg, 'gray')
plt.show()
plt.imshow(img, 'gray')
plt.show()
        
