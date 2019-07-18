#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:56:54 2019

@author: vaeahc
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 200)
f1_x = np.exp(- x ** 2 / 2)
f2_x = np.exp(-np.abs(x))
f3_x = np.concatenate((np.array([0] * 100), np.exp(-np.abs(x[100:]))), axis = 0)

F1_w = np.abs(np.fft.fftshift(np.fft.fft(f1_x)))
F2_w = np.abs(np.fft.fftshift(np.fft.fft(f2_x)))
F3_w = np.abs(np.fft.fftshift(np.fft.fft(f3_x)))
plt.plot(F1_w, 'r')
plt.plot(F2_w, 'g')
plt.plot(F3_w, 'b')
plt.show()