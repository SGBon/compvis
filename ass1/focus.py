import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from scipy import signal

sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])

# determine if an image fragment is in focus
def focus_fragment(frag):
    global laplacian
    fft_frag = np.fft.fft2(frag)
    fshift = np.fft.fftshift(fft_frag)
    mgspec = 20*np.log(np.abs(fshift))
    conv = np.abs(signal.convolve2d(frag,laplacian,mode='same'))
    sum = 0
    avg_coef = conv.shape[0]*conv.shape[1]
    for i in conv:
        for j in i:
            sum += j
    print sum/avg_coef


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Assignment 1 Focus Analysis.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()

    img = cv2.imread(args.imgfile,0)

    w = 15
    dw = w/2
    for i in range(dw,img.shape[0],w):
        for j in range(dw,img.shape[1],w):
            frag = img[i-dw:i+dw,j-dw:j+dw]
            focus_fragment(frag)

    fft_img = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft_img)
    mgspec = 20*np.log(np.abs(fshift))

    conv = np.abs(signal.convolve2d(img,laplacian,mode='same'))

    plt.subplot(121)
    plt.imshow(img,cmap = 'gray')
    plt.subplot(122)
    plt.imshow(conv,cmap='gray')
    plt.show()
