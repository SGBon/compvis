import numpy as np
import cv2
import math

# normalize a matrix
def normalize(mat):
    mat -= np.mean(mat)
    mat /= np.sum(np.abs(mat))
    return mat

# create a filter for LM/RFS
def makefilter(scale,phasex,phasey,pts,sup):
    gx = gauss1d(3*scale,0,pts[0][:],phasex)
    gy = gauss1d(3*scale,0,pts[1][:],phasey)
    f = np.reshape(gx*gy,(sup,sup))
    return normalize(f)

# create 1 dimensional gaussian
def gauss1d(sigma,mean,x,ord):
    x=x-mean
    num = x*x
    variance = sigma**2
    denom = 2*variance
    g = np.exp(-num/denom)/((math.pi*denom)**0.5)
    if ord == 1:
        g = -g*(x/variance)
    elif ord == 2:
        g=g*((num-variance)/(variance**2))
    return g

# create 2d guassian
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

# the exponential part of gaussian
def GaussExp(k,l,mu,cov):
    return math.exp(-(((k-mu)**2 + (l-mu)**2)/(2.0*cov*cov)))

# Laplacian of Gaussian in 2D
def LoG(sigma,n):
    center = n/2
    row = np.linspace(-center,center,n)
    xi,yi = np.meshgrid(row,row)
    kernel = xi
    first = -1/(math.pi*sigma**4)
    for k in range(n):
        for l in range(n):
            kernel[k][l] = first * LoGMid(k,l,sigma) * GaussExp(k,l,0,sigma)
    return kernel

# middle factor of LoG
def LoGMid(k,l,sigma):
    return 1 - ((k**2 + l**2)/(2*sigma**2))

def makeLMfilters():
    # creates LM filter bank
    SUP = 49
    SCALEX = math.sqrt(2.0)**np.array(range(1,4))
    NORIENT = 6

    NROTINV = 12
    NBAR = len(SCALEX)*NORIENT
    NEDGE = len(SCALEX)*NORIENT
    NF=NBAR+NEDGE+NROTINV
    F=np.zeros((NF,SUP,SUP))
    hsup = (SUP-1)/2
    x,y = np.meshgrid(range(-hsup,hsup+1),range(hsup,-hsup-1,-1))
    orgpts = np.array([x.flatten(),y.flatten()])

    count = 0
    for scale in SCALEX:
        for orient in range(0,NORIENT):
            angle = math.pi*orient/NORIENT
            c = math.cos(angle)
            s = math.sin(angle)
            rotpts=np.matmul([[c, -s],[s, c]],orgpts)
            F[count] = makefilter(scale,0,1,rotpts,SUP)
            F[count+NEDGE] = makefilter(scale,0,2,rotpts,SUP)
            count = count+1

    count = NBAR+NEDGE
    SCALES = math.sqrt(2.0)**np.array(range(1,5))
    for scale in SCALES:
        F[count] = normalize(gaussian2d(0,scale,SUP))
        F[count+1] = normalize(LoG(scale,SUP))
        F[count+2] = normalize(LoG(3*scale,SUP))
        count += 3

    return F

def makeRFSfilters():
    # creates LM filter bank
    SUP = 49
    SCALEX = np.array([1,2,4])
    NORIENT = 6

    NROTINV = 2
    NBAR = len(SCALEX)*NORIENT
    NEDGE = len(SCALEX)*NORIENT
    NF=NBAR+NEDGE+NROTINV
    F=np.zeros((NF,SUP,SUP))
    hsup = (SUP-1)/2
    x,y = np.meshgrid(range(-hsup,hsup+1),range(hsup,-hsup-1,-1))
    orgpts = np.array([x.flatten(),y.flatten()])

    count = 0
    for scale in SCALEX:
        for orient in range(0,NORIENT):
            angle = math.pi*orient/NORIENT
            c = math.cos(angle)
            s = math.sin(angle)
            rotpts=np.matmul([[c, -s],[s, c]],orgpts)
            F[count] = makefilter(scale,0,1,rotpts,SUP)
            F[count+NEDGE] = makefilter(scale,0,2,rotpts,SUP)
            count = count+1

    F[NBAR+NEDGE] = normalize(gaussian2d(0,10,SUP))
    F[NBAR+NEDGE+1] = normalize(LoG(10,SUP))

    return F

from scipy import signal

# go through every filter in bank, convolve with image, add response to vector
def vectorFromBank(img,bank):
    vec = []
    for f in bank:
        conv = signal.convolve2d(img,f,mode="same")
        vec.append(np.mean(conv))
    return vec


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Assignment 2 Texture Analysis.')
    parser.add_argument('text1', help='First texture file')
    parser.add_argument('text2', help='Second texture file')
    args = parser.parse_args()

    text1 = cv2.imread(args.text1)
    text1b,text1g,text1r = cv2.split(text1)
    text2 = cv2.imread(args.text2)
    text2b,text2g,text2r = cv2.split(text2)

    # empty vectors
    t1_vec = np.append([],[])
    t2_vec = np.append([],[])

    # get variations of "redness", "blueness", "greeness" in images
    t1_vec = np.append(t1_vec,(np.std(text1r),np.std(text1g),np.std(text1b)))
    t2_vec = np.append(t2_vec,(np.std(text2r),np.std(text2g),np.std(text2b)))

    # get means of redness, etc.
    t1_vec = np.append(t1_vec,(np.mean(text1r),np.mean(text1g),np.mean(text1b)))
    t2_vec = np.append(t2_vec,(np.mean(text2r),np.mean(text2g),np.mean(text2b)))

    print t1_vec
    print t2_vec

    text1gray = cv2.cvtColor(text1,cv2.COLOR_BGR2GRAY)
    text2gray = cv2.cvtColor(text2,cv2.COLOR_BGR2GRAY)

    LMF = makeLMfilters()
    RFSF = makeRFSfilters()
    print " ----------------- LMF ----------------"
    print vectorFromBank(text1gray,LMF)
    print "------------------ RFSF --------------------"
    print vectorFromBank(text1gray,RFSF)
