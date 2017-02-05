import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from scipy import signal

# laplacian is a high pass filter, so we're using it hear to essentially take everything
# that isn't in focus out of the image (because low focus == low frequency)
laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])

# get an "in focus" score from the fragment
def score_fragment(frag):
    global laplacian
    conv = np.abs(signal.convolve2d(frag,laplacian,mode='same'))
    sum = 0
    n = conv.shape[0]*conv.shape[1]
    for i in conv:
        for j in i:
            sum += j
    avg = sum/n
    print avg
    return avg

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Assignment 1 Focus Analysis.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()

    img = cv2.imread(args.imgfile,0)

    w = 15
    dw = w/2
    threshold = 40
    elapsed = time.clock()
    for i in range(dw,img.shape[0],w):
        for j in range(dw,img.shape[1],w):
            frag = img[i-dw:i+dw,j-dw:j+dw]
            score = score_fragment(frag)
            # if fragment score is passes some threshold, then add it to in focus
    elapsed = (time.clock() - elapsed) * 1000

    print "Focus analysis took %f milliseconds" % elapsed

    conv = np.abs(signal.convolve2d(img,laplacian,mode='same'))

    plt.subplot(121)
    plt.imshow(img,cmap = 'gray')
    plt.subplot(122)
    plt.imshow(conv,cmap='gray')
    plt.show()
