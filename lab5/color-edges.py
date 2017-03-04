import numpy as np
import cv2
from scipy import signal
import math
import matplotlib.pyplot as plt

sobel_x = [[-1,-2,-1],[0,0,0],[1,2,1]]
sobel_y = [[-1,0,1],[-2,0,2],[-1,0,1]]

cmap = [[0,177,215],
        [13,85,221],
        [1,37,187],
        [184,134,203],
        [244,36,148],
        [255,32,0],
        [253,134,8],
        [255,166,2],
        [253,205,7],
        [255,250,0],
        [133,207,0],
        [2,184,1]]

if __name__ == '__main__':
    img = cv2.imread("lena.png",0)
    edges = cv2.Canny(img,200,250)
    dx = signal.convolve2d(img,sobel_x,mode="same")
    dy = signal.convolve2d(img,sobel_y,mode="same")
    angles = np.arctan2(dy,dx)
    angles = angles + math.pi # move angles forward by pi so no negative values

    emap = edges / np.max(edges) # map of pixels that are on edges
    angles = np.multiply(emap,angles) # remove angles not on edges

    x,y = img.shape
    final = np.zeros((x,y,3),np.uint8) # final image
    bins = np.digitize(angles,np.linspace(0,math.pi*2,12))
    bins = bins - 1
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if emap[i][j] != 0: # make sure the current pixel is an edge
                final[i][j] = cmap[bins[i][j]]

    # create histogram
    histo = np.zeros(12,int)
    for i in range(0,bins.shape[0]):
        for j in range(0,bins.shape[1]):
            if emap[i][j] != 0:
                histo[bins[i][j]] = histo[bins[i][j]] + 1

    plt.subplot(121)
    plt.title("edge directions")
    plt.imshow(final)
    plt.subplot(122)
    plt.bar(range(len(histo)),histo)
    plt.title("histogram of edges")
    plt.xlabel("Bucket")
    plt.ylabel("Frequency")
    plt.show()
