#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:59:40 2019

@author: qifan luo
"""

import pylab as plt
import matplotlib.image as mpimg
import numpy as np

def hist (rgb):
    m,n = rgb.shape
    h = np.zeros((256,), dtype=np.float)
    for i in range(m):
        for j in range(n):
            h[rgb[i, j]]+=1
    return np.array(h)

img = np.array(mpimg.imread('/Users/fantaaaa/Desktop/b1.png')*255.0)

bins = np.arange(257)
r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]

#plot rgb
plt.figure(0)
plt.title('Red histogram')
r_hist = hist (np.uint8(r))
for i in range(256):
    plt.bar(i,r_hist[i],color='r')

plt.figure(1)
plt.title('Green histogram')
g_hist = hist (np.uint8(g))
for i in range(256):
    plt.bar(i,g_hist[i],color='g')
plt.figure(2)
plt.title('Blue histogram')
b_hist = hist (np.uint8(b))
for i in range(256):
    plt.bar(i,b_hist[i],color='b')

#show gray image
img = np.array(r*0.3) + np.array(g*0.59) + np.array(b*0.11)
img = np.uint8(img)
plt.figure(3)
plt.imshow(img,cmap='gray')
plt.title('gray image')


#show gray histogram
plt.figure(4)
plt.title('gray histogram')
gray_hist = hist (img)
for i in range(256):
    plt.bar(i,gray_hist[i],color='black')

#pdf
pdf = gray_hist/img.size
plt.figure(5)
plt.plot(pdf)
plt.title('pdf')


#cdf
plt.figure(6)
cdf=pdf.cumsum()
plt.plot(cdf)
plt.title('cdf')


#histogram equalization
equalize=255*np.array(cdf[np.uint8(img)])
plt.figure(7)
plt.title('histogram equalization')
plt.hist(equalize.flatten(), bins=bins)


#new image
plt.figure(8)
plt.title('new image')
plt.imshow(equalize,cmap='gray')






