#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:22:26 2019

@author: vaeahc
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('4.2.03.tiff', 0)
row = np.zeros((1, im.shape[1]))
im = np.row_stack((row, im, row))
col = np.zeros((im.shape[0], 1))
im = np.column_stack((col, im, col))

sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

im_out = np.zeros((im.shape[0], im.shape[1]))
for i in range(1, im_out.shape[0] - 1):
    for j in range(1, im_out.shape[1] - 1):
        im_out[i][j] = np.abs(np.sum(sx * im[i-1:i+2, j-1:j+2])) + np.abs(np.sum(sy * im[i-1:i+2, j-1:j+2]))

im_out = im_out[1:-1, 1:-1]
for i in range(im_out.shape[0]):
    for j in range(im_out.shape[1]):
        if im_out[i][j] <= 300:
            im_out[i][j] = 0
        else:
            im_out[i][j] = 255
    
plt.imshow(im_out, 'gray')
plt.show()


