import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Assignment 1 Focus Analysis.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()

    img = cv2.imread(args.imgfile,0)

    fft_img = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft_img)
    mgspec = 20*np.log(np.abs(fshift))

    print mgspec

    plt.subplot(121)
    plt.imshow(img,cmap = 'gray')
    plt.subplot(122)
    plt.imshow(mgspec,cmap='gray')
    plt.show()
