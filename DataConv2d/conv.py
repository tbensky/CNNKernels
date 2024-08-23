#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import torch
import sys



#https://stackoverflow.com/questions/2448015/2d-convolution-using-python-and-numpy
def convolution2d(image, kernel, bias):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) + bias
    return new_image


#img = plt.imread('square_pulse.jpg')
#img = plt.imread('gaussian.jpg')
img = plt.imread('sinusoid.jpg')


red = img[:,:,1]
red = red - red.min()
red = red / red.max()
print(red.max(),red.min())
print(red.shape)



plt.imshow(red,aspect='equal',cmap='gray')
plt.show()

#conv03 gives an interesting result
#kimg = plt.imread('conv01_smaller.png')
#kimg = plt.imread('conv02_smaller.png')


k = torch.load("../Keep/23Aug/kernel_00411_00.pt")
kimg =  k.permute(1, 2, 0) 
kred = kimg.tolist()

plt.imshow(kred,aspect='equal',cmap='gray')
plt.savefig("kernel.png",dpi=300)
plt.show()
kred = np.array(kred).reshape((20,20))

ax1 = plt.subplot(2, 1, 1)
plt.imshow(red,cmap='gray')

#c = signal.convolve2d(red, kred)
c = convolution2d(red,kred,0)
ax2 = plt.subplot(2,1,2)
plt.imshow(c,aspect='equal',cmap='gray')
ax2.sharex(ax1)
plt.savefig("result_conv2d.png",dpi=300)
plt.show()