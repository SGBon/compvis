# filter.py
import argparse
import cv2
import numpy as np
from scipy import signal
import math
from matplotlib import pyplot as plt

def gaussian2d(mu,cov,n):
    center = n/2
    sum = 0
    row = np.linspace(-center,center,n)
    xi,yi = np.meshgrid(row,row)
    kernel = yi

    first = 1.0/(math.sqrt(2.0*math.pi)*cov*cov) # the non-varying coefficient
    for k in range(n):
        for l in range(n):
            next = first * GaussExp(xi[k][l],yi[k][l],mu,cov)
            sum = sum + next
            kernel[k][l] = next
    kernel = kernel / sum
    return kernel;

def GaussExp(k,l,mu,cov):
    return math.exp(-(((k-mu)**2 + (l-mu)**2)/(2.0*cov*cov)))

def filter(img,task):
    # Complete this method according to the tasks listed in the lab handout.
    if task == 1:
        kernel = gaussian2d(0,5,5)
    elif task == 2:
        kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    elif task == 3:
        kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    elif task == 4:
        kernel = np.zeros([5,5])
    elif task == 5:
        kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    return signal.convolve2d(img,kernel)

def process_img1(imgfile,task):
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # You should implement your functionality in filter function
    filtered_img = filter(img,task).astype(np.uint8)
    print img,filtered_img

    cv2.imshow('Input image',img)
    cv2.imshow('Filtered image',filtered_img)

    print 'Press any key to proceed'
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_img2(imgfile,task):
    print 'Opening ', imgfile
    img = cv2.imread(imgfile)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # You should implement your functionality in filter function
    filtered_img = filter(img,task)

    # You should implement your functionality in filter function

    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(filtered_img)
    plt.title('Filtered image')
    plt.xticks([]), plt.yticks([])

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Lab 2.')
    parser.add_argument('--use-plotlib', action='store_true', help='If specified uses matplotlib for displaying images.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()

    task = 1

    if args.use_plotlib:
        process_img2(args.imgfile,task)
    else:
        process_img1(args.imgfile,5)
