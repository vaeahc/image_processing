#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:42:49 2019

@author: vaeahc
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('IMG_0381.tif', 0)
img2 = cv2.imread('IMG_1136.tif', 0)

plt.imshow(img1, 'gray')
plt.show()
plt.imshow(img2, 'gray')
plt.show()

img1_t = np.fft.fftshift(np.fft.fft2(img1))
img1_t_A = np.abs(img1_t)
img1_t_phase = np.angle(img1_t)

img2_t = np.fft.fftshift(np.fft.fft2(img2))
img2_t_A = np.abs(img2_t)
img2_t_phase = np.angle(img2_t)

img3 = np.zeros(img1.shape, dtype = complex) #1 A, 2 phase
img3_real = img1_t_A * np.cos(img2_t_phase)
img3_i = img1_t_A * np.sin(img2_t_phase)
img3.real = img3_real
img3.imag = img3_i
img3 = np.abs(np.fft.ifft2(np.fft.ifftshift(img3)))
plt.imshow(img3, 'gray')
plt.show()

img4 = np.zeros(img1.shape, dtype = complex) #1 phase, 2 A
img4_real = img2_t_A * np.cos(img1_t_phase)
img4_i = img2_t_A * np.sin(img1_t_phase)
img4.real = img4_real
img4.imag = img4_i
img4 = np.abs(np.fft.ifft2(np.fft.ifftshift(img4)))
plt.imshow(img4, 'gray')
plt.show()




 

