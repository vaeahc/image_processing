#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:53:47 2019

@author: vaeahc
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

im = cv2.imread('IMG_2126.PNG', 0)

row = np.zeros((1, im.shape[1]))
im = np.row_stack((row, im, row))
col = np.zeros((im.shape[0], 1))
im = np.column_stack((col, im, col))

rod = 5
K = 100
lamda = 0.1
im_out = copy.deepcopy(im)

def g(grad, k):
    return np.exp(-(grad / k) ** 2)

for k in range(rod):
    for i in range(1, im.shape[0] - 1):
        for j in range(1, im.shape[1] - 1):
            grad_N = (im[i - 1][j] - im[i][j]) / 2
            grad_S = (im[i + 1][j] - im[i][j]) / 2
            grad_W = (im[i][j - 1] - im[i][j]) / 2
            grad_E = (im[i][j + 1] - im[i][j]) / 2
            
            im_out[i][j] -= lamda * (g(grad_N, K) * grad_N + g(grad_S, K) * grad_S + g(grad_W, K) * grad_W + g(grad_E, K) * grad_E)

plt.imshow(im_out[1:-1, 1:-1], 'gray') 
plt.show()           

            
            
            
