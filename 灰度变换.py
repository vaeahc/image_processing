#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:55:51 2019

@author: vaeahc
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def fupian(im):
    #用于观察过黑的图片，凸显暗背景下的亮物体
    return 1 - im

def log(im, v):
    #低灰度部分扩展，高灰度部分压缩；底数越大，低灰度部分扩展越强。用于强调图像低灰度部分。
    return np.log(1 + v * im) / np.log(1 + v)

def draw_log():
    
    x = np.linspace(0, 1, 100)
    V = [1, 10, 30, 100, 200]
    color = ['b', 'g', 'r', 'c', 'm']
    for i in range(len(V)):
        y = np.log(1 + V[i] * x) / np.log(1 + V[i])
        c = color[i]
        plt.plot(x, y, c, label = 'v = %d' % V[i])
    plt.title('log curve')
    plt.legend()
    plt.xlabel('I(x, y)')
    plt.ylabel("I'(x, y)")
    plt.axis([0, 1, 0, 1])
    plt.show()
    
def gama(im, g):
    
    return np.power(im, g)

def draw_gama():
    
    x = np.linspace(0, 1, 100)
    gama = [0.1, 0.2, 0.4, 2.5, 5.0, 10.0]
    color = ['b', 'g', 'r', 'c', 'm', 'y']
    for i in range(len(gama)):
        y = np.power(x, gama[i])
        c = color[i]
        plt.plot(x, y, c, label = 'gama = %f' % gama[i])
    plt.title('gama curve')
    plt.legend()
    plt.xlabel('I(x, y)')
    plt.ylabel("I'(x, y)")
    plt.axis([0, 1, 0, 1])
    plt.show()

#灰度拉伸    
def huidu_las(im):
    #0处理
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            
            if im[i][j] == 0:
                im[i][j] += 1e-4
                
    m = 1 / 2 * (np.min(im) + np.max(im))
    e1 = np.log(1 / 0.05 - 1) / np.log(m / np.min(im))    
    e2 = np.log(1 / 0.95 - 1) / np.log(m / np.max(im))
    e = (e1 + e2) / 2
    
    return 1 / (1 + np.power(m / im, e))

def cal_hist(img):
    #img为0-255
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    plt.plot(hist)
    plt.xlabel('灰度级')
    plt.ylabel('频率')
    plt.title('灰度直方图')
    plt.show()
       
    
        
if __name__ == '__main__':
    
    img = cv2.imread('lena512color.tiff', 0)
    '''
    img_1 = fupian(img)
    plt.imshow(img_1, 'gray')
    plt.show()
    '''
    '''
    img_2 = log(img, 1)
    plt.imshow(img_2, 'gray')
    plt.show()
    '''
    '''
    draw_log()
    '''
    '''
    img_3 = gama(img, 5)
    plt.imshow(img_3, 'gray')
    plt.show()
    '''
    '''
    draw_gama()
    '''
    '''
    img_4 = huidu_las(img)
    plt.imshow(img_4, 'gray')
    plt.show()
    '''
    '''
    cal_hist(img)
    '''
    #直方图均衡化

    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    hist /= (img.shape[0] * img.shape[1])
    hist_tr = np.zeros(np.shape(hist))
    for i in range(256):
        
        if i == 0:
            hist_tr[0][0] = hist[0][0]
        else:
            hist_tr[i][0] = hist_tr[i - 1][0] + hist[i][0]
    
    img_out = np.zeros(np.shape(img))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            img_out[i][j] = hist_tr[img[i][j]]
    img_out = (img_out * 255).astype(np.uint8)
    
    hist_1 = cv2.calcHist([img], [0], None, [256], [0, 255])    
    hist_2 = cv2.calcHist([img_out], [0], None, [256], [0, 255])
    plt.subplot(2, 1, 1)
    plt.plot(hist_1)
    plt.subplot(2, 1, 2)
    plt.plot(hist_2)
    plt.show()


    
    
    
    
                
                
                

    
    
    
    
