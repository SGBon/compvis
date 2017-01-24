import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

sobel_X = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

img = cv2.imread("lena.png",0)

print "filtering"
filtered = signal.convolve2d(img,img,mode='same')
print "filtered"

plt.figure(figsize=(5,5))
plt.imshow(filtered,'gray')
