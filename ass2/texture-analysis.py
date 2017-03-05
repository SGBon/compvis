import numpy as np
from scipy import signal
from scipy import spatial
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
    # creates RFS filter bank
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

def makeSfilters():
    # creates S filter bank
    NF = 13
    SUP = 49
    F = np.zeros((NF,SUP,SUP))

    F[0] = makeSfilter_helper(SUP,2,1)
    F[1] = makeSfilter_helper(SUP,4,1)
    F[2] = makeSfilter_helper(SUP,4,2)
    F[3] = makeSfilter_helper(SUP,6,1)
    F[4] = makeSfilter_helper(SUP,6,2)
    F[5] = makeSfilter_helper(SUP,6,3)
    F[6] = makeSfilter_helper(SUP,8,1)
    F[7] = makeSfilter_helper(SUP,8,2)
    F[8] = makeSfilter_helper(SUP,8,3)
    F[9] = makeSfilter_helper(SUP,10,1)
    F[10] = makeSfilter_helper(SUP,10,2)
    F[11] = makeSfilter_helper(SUP,10,3)
    F[12] = makeSfilter_helper(SUP,10,4)
    return F

def makeSfilter_helper(sup,sigma,tau):
    hsup = (sup-1)/2
    x,y = np.meshgrid(range(-hsup,hsup+1),range(hsup,-hsup-1,-1))
    r = (x*x+y*y)**0.5
    f = np.cos(r*(math.pi*tau/sigma))*np.exp(-(r*r)/(2*sigma*sigma))
    return normalize(f)

# go through every filter in bank, convolve with image, add responses to vector
def vectorFromBank(img,bank):
    vec = []
    for f in bank:
        conv = signal.convolve2d(img,f,mode="same")
        vec.append(np.mean(conv))
        vec.append(np.std(conv))
    return vec

LMF = makeLMfilters()
RFSF = makeRFSfilters()
SF = makeSfilters()

# vectorizes a texture
def vectorize(image):
    global LMF
    global RFSF
    ib,ig,ir = cv2.split(image)
    igray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    vec = np.append([],[]) # create empty vector

    #vec = np.append(vec,(np.std(ir),np.std(ig),np.std(ib)))
    #vec = np.append(vec,(np.mean(ir),np.mean(ig),np.mean(ib)))
    vec = np.append(vec,vectorFromBank(igray,LMF))
    vec = np.append(vec,vectorFromBank(igray,RFSF))
    vec = np.append(vec,vectorFromBank(igray,SF))
    return vec


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Assignment 2 Texture Analysis.')
    parser.add_argument('text1', help='First texture file')
    parser.add_argument('text2', help='Second texture file')
    args = parser.parse_args()

    text1 = cv2.imread(args.text1)
    t1_vec = vectorize(text1)
    #print t1_vec

    text2 = cv2.imread(args.text2)
    t2_vec = vectorize(text2)
    #print t2_vec

    # calculate some distances
    euclid = spatial.distance.euclidean(t1_vec,t2_vec)
    cosine = spatial.distance.cosine(t1_vec,t2_vec)
    cheb = spatial.distance.chebyshev(t1_vec,t2_vec)
    braycurt = spatial.distance.braycurtis(t1_vec,t2_vec)
    canb = spatial.distance.canberra(t1_vec,t2_vec)
    corr = spatial.distance.correlation(t1_vec,t2_vec)

    print euclid,cosine,cheb,braycurt,canb,corr
