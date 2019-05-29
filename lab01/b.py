#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 22:55:37 2019

@author: qifan luo
"""
import numpy as np 
import pylab as plt
import matplotlib.image as mpimg

def hist (rgb):
    m,n = rgb.shape
    h = np.zeros((256,), dtype=np.float)
    for i in range(m):
        for j in range(n):
            h[rgb[i, j]]+=1
    return np.array(h)

def binaray(gray,thresh):  
    final_img = gray.copy()
    final_img[gray > thresh] = 255
    final_img[gray <= thresh] = 0
    return final_img
      
def otsu(gray):
    mean_weigth = 1.0/gray.size
    his = hist(gray)
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    plt.title('variance')
    for t in np.arange(256):
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth
        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        value = Wb * Wf * (mub - muf) ** 2
        plt.bar(t,value)
        if value > final_value:
            final_thresh = t
            final_value = value
    plt.show()
    print ('variance:',final_value)
    print ('thresh:',final_thresh)
    final_img = gray.copy()
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img


imga = np.uint8(mpimg.imread('/Users/fantaaaa/Desktop/b2_a.png')*255.0)
imgb = np.uint8(mpimg.imread('/Users/fantaaaa/Desktop/b2_b.png')*255.0)
imgc = np.uint8(mpimg.imread('/Users/fantaaaa/Desktop/b2_c.png')*255.0)

bins = np.arange(257)

#binaray picuture
plt.figure(0)
plt.title('a, thresh=55')
plt.imshow(binaray(imga,55),cmap='gray')
plt.figure(1)
plt.title('b, thresh=124')
plt.imshow(binaray(imgb,124),cmap='gray')
plt.figure(2)
plt.title('c, thresh=90')
plt.imshow(binaray(imgc,90),cmap='gray')


#show histogram
plt.figure(3)
plt.title('a, histogram')
a_hist = hist (np.uint8(imga))
for i in range(256):
    plt.bar(i,a_hist[i],color='black')
plt.figure(4)
plt.title('b, histogram')
b_hist = hist (np.uint8(imgb))
for i in range(256):
    plt.bar(i,b_hist[i],color='black')
plt.figure(5)
plt.title('c, histogram')
c_hist = hist (np.uint8(imgc))
for i in range(256):
    plt.bar(i,c_hist[i],color='black')

#Otsu
plt.figure(6)
plt.imshow(otsu(imga),cmap='gray')
plt.figure(7)
plt.imshow(otsu(imgb),cmap='gray')
plt.figure(8)
plt.imshow(otsu(imgc),cmap='gray')
