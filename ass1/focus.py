import numpy as np
import cv2
from scipy import signal

# laplacian is a high pass filter, so we're using it hear to essentially take everything
# that isn't in focus out of the image (because low focus is usually low frequency)
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Assignment 1 Focus Analysis.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()

    img = cv2.imread(args.imgfile,0)

    w = 15
    dw = w/2
    threshold = 55
    in_focus = []
    print img.shape
    elapsed = time.clock()
    for i in range(dw,img.shape[0],w):
        for j in range(dw,img.shape[1],w):
            frag = img[i-dw:i+dw,j-dw:j+dw]
            score = score_fragment(frag)
            # if fragment score is passes some threshold, then add it to in focus
            if score >= threshold:
                in_focus.append((i-dw,i+dw,j-dw,j+dw))
    elapsed = (time.clock() - elapsed) * 1000

    print "Focus analysis took %f milliseconds" % elapsed

    conv = np.abs(signal.convolve2d(img,laplacian,mode='same'))

    ax = plt.subplot(121)
    plt.imshow(img,cmap = 'gray')
    for i in in_focus:
        rect = patches.Rectangle((i[2],i[0]),i[3] - i[2],i[1]-i[0],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.subplot(122)
    plt.imshow(conv,cmap='gray')
    plt.show()
