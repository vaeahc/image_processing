#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:47:24 2019

@author: vaeahc
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy

im = cv2.imread('IMG_2126.PNG', 0)
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

row = np.zeros((1, im.shape[1]))
im = np.row_stack((row, im, row))
col = np.zeros((im.shape[0], 1))
im = np.column_stack((col, im, col))

im_x = np.zeros(np.shape(im))
im_y = np.zeros(np.shape(im))
im_xx = np.zeros(np.shape(im))
im_yy = np.zeros(np.shape(im))
im_xy = np.zeros(np.shape(im))
im_out = np.zeros(np.shape(im))

for i in range(1, im_x.shape[0] - 1):
    for j in range(1, im_x.shape[1] - 1):
        im_x[i][j] = np.sum(sx * im[i - 1: i + 2, j - 1: j + 2])
        im_y[i][j] = np.sum(sy * im[i - 1: i + 2, j - 1: j + 2])
    
for i in range(1, im_xx.shape[0] - 1):
    for j in range(1, im_xx.shape[1] - 1):
        im_xx[i][j] = np.sum(sx * im_x[i - 1: i + 2, j - 1: j + 2])
        im_yy[i][j] = np.sum(sy * im_y[i - 1: i + 2, j - 1: j + 2])
        im_xy[i][j] = np.sum(sy * im_x[i - 1: i + 2, j - 1: j + 2])
        
im_n = np.zeros((np.shape(im)))

for i in range(1, im_n.shape[0]):
    for j in range(1, im_n.shape[1]):
        im_n[i][j] = (im_xx[i][j] * im_y[i][j] ** 2 - 2 * im_xy[i][j] * im_x[i][j] * im_y[i][j]  + im_yy[i][j] * im_x[i][j] ** 2) / (im_x[i][j] ** 2 + im_y[i][j] ** 2)
        
M = 6
t = 0.1
im_out = copy.deepcopy(im)
for i in range(M):
    im_out -= t * im_n
    
plt.imshow(im_out, 'gray')
plt.show()
    
    




