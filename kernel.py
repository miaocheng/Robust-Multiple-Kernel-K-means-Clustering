
#----------------------------------------------------------------------------------------------------
# Name:        kernel.py
#
# This code implements the kernel based calculation methods.
#
# Coded by:      Miao Cheng
# E-mail: mewcheng@gmail.com
# Current Version: v0.91
# Created Date:     11/13/2018
# All Rights Reserved
#----------------------------------------------------------------------------------------------------

import numpy as np
import scipy.linalg as la
from cala import *


def Kernel(X, Y, ktype):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    
    if ktype == 'Gaussian':
        D = eudist(X, Y, False)
        t = 1
        tmp = 2*(t**2)
        #tmp = sigma
        D = -D / tmp
        K = np.exp(D)
        del D
        
    elif (ktype == 'Cosine'):
        K = cosdist(X, Y)
        
    elif (ktype == 'Sigmoid'):
        D = eudist(X, Y, False)
        D = - D
        K = 1 + np.exp(D)
        K = float(1) / K
        del D
        
    elif ktype == 'Sinc':
        K = np.zeros((xSam, ySam))
        
        for i in range(xSam):
            for j in range(ySam):
                tmp = X[:, i] - Y[:, j]
                tmp = np.sum(tmp, axis=0)
                tmp = np.sinc(tmp)
                K[i, j] = tmp
                
    elif ktype == 'Line':
        K = np.dot(np.transpose(X), Y)   
        
    return K


def getKernel(X, Y, **kwargs):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    assert xDim == yDim, 'The dimensionality of two data sets is not identical !'
    
    # +++++ Get the parameter in kernels +++++
    if 'ktype' not in kwargs:
        ktype = 'Gaussian'
    else:
        ktype = kwargs['ktype']
        
    if 't' not in kwargs:
        t = 1
    else:
        t = kwargs['t']
    
    if ktype == 'Line':
        K = np.dot(np.transpose(X), Y)
    
    if (ktype == 'Gaussian'):
        D = eudist(X, Y, False)
        tmp = 2*(t**2)
        D = - D / tmp
        K = np.exp(D)
        del D
        
    elif (ktype == 'Cosine'):
        K = cosdist(X, Y)
        
    elif (ktype == 'Sine'):
        K = np.zeros((xSam, ySam))
        
        for i in range(xSam):
            for j in range(ySam):
                tmp = X[:, i] - Y[:, j]
                tmp = np.sum(tmp, axis=0)
                tmp = np.sinc(tmp)
                K[i, j] = tmp
                
        del tmp
        
    elif (ktype == 'Sigmoid'):
        D = eudist(X, Y, False)
        D = - D
        K = 1 + np.exp(D)
        K = float(1) / K
        del D
        
    return K
    
    
def sigmod(X):
    xDim, xSam = np.shape(X)
    
    Y = np.zeros((xDim, xSam))
    for i in range(xDim):
        for j in range(xSam):
            tmp = X[i, j]
            tmp = 1 + np.exp(- tmp)
            tmp = float(1) / tmp
            Y[i, j] = tmp
    
    return Y


