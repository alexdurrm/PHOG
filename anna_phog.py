import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt

'''
original matlab code at https://www.robots.ox.ac.uk/~vgg/research/caltech/phog.html
'''

def anna_phog(Img, bin, angle, L, roi):
    '''
     anna_PHOG Computes Pyramid Histogram of Oriented Gradient over a ROI.

     [BH, BV] = anna_PHOG(Img,BIN,ANGLE,L,ROI) computes phog descriptor over a ROI.

     Given and image Img, phog computes the Pyramid Histogram of Oriented Gradients
     over L pyramid levels and over a Region Of Interest

    IN:
        Img- Images numpy array of size MxN (Color or Gray)
        bin - Number of bins on the histogram
        angle - 180 or 360
       L - number of pyramid levels
       roi - Region Of Interest (ytop,ybottom,xleft,xright)

    OUT:
        p - pyramid histogram of oriented gradients
    '''
    if Img.shape[2] == 3:
        G = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    else:
        G = Img

    if np.sum(G) > 100:
        # apply automatic Canny edge detection using the computed median
        sigma = 0.33
        v = np.median(G)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        E = cv2.Canny(G,lower,upper) #high and low treshold
        GradientX, GradientY = np.gradient(G)
        GradientYY = np.gradient(GradientY, axis=1)

        Gr = np.sqrt(np.square(GradientX)+np.square(GradientY))
        index = GradientX == 0
        GradientX[index] = 1e-5 #maybe another value

        YX = GradientY*GradientX
        if angle == 180: A = ((np.arctan(YX)+(np.pi/2))*180)/np.pi
        if angle == 360: A = ((np.arctan2(GradientY,GradientX)+np.pi)*180)/np.pi

        bh, bv = anna_BinMatrix(A,E,Gr,angle,bin)

    else:
        bh = np.zeros(Img.shape)
        bv = np.zeros(Img.shape)

    bh_roi = bh[roi[0]:roi[1], roi[2]:roi[3]]
    bv_roi = bv[roi[0]:roi[1], roi[2]:roi[3]]

    p = anna_PhogDescriptor(bh_roi,bv_roi,L,bin)
    return p


def anna_BinMatrix(A,E,G,angle,bin):
    '''
    anna_BINMATRIX Computes a Matrix (bm) with the same size of the image where
    (i,j) position contains the histogram value for the pixel at position (i,j)
    and another matrix (bv) where the position (i,j) contains the gradient
    value for the pixel at position (i,j)

    IN:
    	A - Matrix containing the angle values
    	E - Edge Image
        G - Matrix containing the gradient values
    	angle - 180 or 360
        bin - Number of bins on the histogram

    OUT:
    	bm - matrix with the histogram values
        bv - matrix with the gradient values (only for the pixels belonging to and edge)
    '''
    n, contorns = cv2.connectedComponents(E, connectivity=8)

    X = E.shape[1]
    Y = E.shape[0]
    bm = np.zeros(shape=(Y,X))
    bv = np.zeros(shape=(Y,X))

    nAngle = angle/bin

    for i in range(n):
        posY, posX = np.where(contorns==i)
        for j in range(posY.shape[0]):
            pos_x = posX[j]
            pos_y = posY[j]

            b = np.ceil(A[pos_y,pos_x]/nAngle)
            if b==0: bin=1
            if G[pos_y,pos_x]>0:
                bm[pos_y,pos_x] = b
                bv[pos_y,pos_x] = G[pos_y,pos_x]

    return (bm, bv)



def anna_PhogDescriptor(bh,bv,L,bin):
    '''
     anna_PHOGDESCRIPTOR Computes Pyramid Histogram of Oriented Gradient over a ROI.

    IN:
        bh - matrix of bin histogram values
        bv - matrix of gradient values
       L - number of pyramid levels
       bin - number of bins

    OUT:
        p - pyramid histogram of oriented gradients (phog descriptor)
    '''
    p = np.array([])
    #level 0
    for b in range(bin):
        ind = bh==b
        p = np.append(p, np.sum(bv[ind]))

    #higher levels
    for l in range(1, L+1):
        x = int(np.trunc(bh.shape[1]/(2**l)))
        y = int(np.trunc(bh.shape[0]/(2**l)))
        for xx in range(0, bh.shape[1]-x+1, x):
            for yy in range(0, bh.shape[0]-y+1, y):
                print(l)
                bh_cella = bh[yy:yy+y, xx:xx+x]
                bv_cella = bv[yy:yy+y, xx:xx+x]

                for b in range(bin):
                    ind = bh_cella==b
                    p = np.append(p, np.sum(bv_cella[ind], axis=0))

    if np.sum(p)!=0:
        p = p/np.sum(p)

    return p
