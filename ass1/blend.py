import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Assignment 1 Focus Analysis.')
    parser.add_argument('imgfile1', help='Image file')
    parser.add_argument('imgfile2',help='Second image file')
    parser.add_argument('a',type=float,help='Blending coefficient')
    args = parser.parse_args()

    img1 = cv2.imread(args.imgfile1,0)
    img2 = cv2.imread(args.imgfile2,0)
    alpha = args.a
    img1 = img1 * alpha
    img2 = img2 * (1-alpha)
    final = img1 + img2

    plt.imshow(final,cmap="gray")
    plt.show()
