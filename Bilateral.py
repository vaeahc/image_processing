#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:07:19 2019

@author: vaeahc
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Bilateral(A):
    #5 * 5双边滤波
    #定义域标准差
    sigma_d = 4
    #值域标准差
    sigma_r = 0.1
    
    w = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            
            c = A[5][5]
            w[i][j] = np.exp(-((i - 5) ** 2 + (j - 5) ** 2) / (2 * sigma_d ** 2) - (c - A[i][j]) ** 2 / (2 * sigma_r ** 2))
    w /= np.sum(w)
    
    return w

img = cv2.imread('lena512color.tiff', 0) / 255

im_out = np.zeros(np.shape(img))
img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_DEFAULT)
for i in range(im_out.shape[0]):
    for j in range(im_out.shape[1]):
        
        A = img[i: i + 11, j : j + 11]
        im_out[i][j] = np.sum(A * Bilateral(A))

plt.imshow(im_out, 'gray')
plt.show()

'''
im = cv2.GaussianBlur(img, (11, 11), 4)
plt.imshow(im, 'gray')
plt.show()
'''
            
            
            