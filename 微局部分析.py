#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:12:50 2019

@author: vaeahc
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('tt.jpg', 0)
plt.imshow(im, 'gray')
plt.show()

Gaussian = np.zeros((21, 21))
sigma = 1
for i in range(21):
    for j in range(21):
        Gaussian[i][j] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((i - 10) ** 2 + (j - 10) ** 2) / (2 * sigma ** 2))
Gaussian /= np.sum(Gaussian)

im_test = im[59:80, 229:250] * Gaussian
im_test_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im_test))))
plt.imshow(im_test_fft, 'gray')
plt.show()
