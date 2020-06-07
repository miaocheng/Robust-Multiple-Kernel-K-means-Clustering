
# --------------------------------------------------------------------------------------------------
# This python file contains the implementation of the RMKKM algorithm.
# Reference:
# 
# L. Du, P. Zhou, L. Shi, H. Wang, M. Fan, W. Wang, Y. D. Shen, Robust Multiple Kernel K-means
# Using l_2,1 Norm, IJCAI 2015.
# Coded by Miao Cheng
# Email: miao_cheng@outlook.com
# Date: 2020-03-14
# --------------------------------------------------------------------------------------------------
import numpy as np

from cala import *
from kernel import *


def predCluster(Z):
    nSam, c = np.shape(Z)
    assert c > 0, 'The obtained Z is incorrect !'
    
    labels = []
    for i in range(nSam):
        tmp = Z[i, :]
        tid = seArr(1, tmp)
        
        labels.append(tid[0])
        
    labels = np.array(labels) + 1
    
    return labels


def rmkkm(Fea, c, gamma, nIter):
    nFea = len(Fea)
    tmx = Fea[0]
    nDim, nSam = np.shape(tmx)
    
    K = []
    for i in range(nFea):
        tmx = Fea[i]
        tmk = Kernel(tmx, tmx, 'Gaussian')
        K.append(tmk)
        
    # +++++ Initialize Z, w, D +++++
    Z = np.zeros((nSam, c))
    assert c < nSam, 'The amount of clusters are insuitable !'
    for i in range(c):
        ind = np.array(range(nSam))
        np.random.shuffle(ind)
        tid = ind[0]
        
        Z[tid, i] = 1
        
    w = np.ones((nFea, 1)).reshape(nFea, )
    w = (float(1) / nFea) * w
    D = np.eye(nSam)
    
    obj = 1e7
    # +++++ Repeatation +++++
    for ii in range(nIter):
        tmp = np.dot(D, Z)      # nSam * c
        s = np.sum(tmp, axis=0)
        s = np.transpose(repVec(s, nSam))
        A = tmp / s
        
        # +++++ Update Z +++++
        # +++++ Check the determined value of z_ij +++++
        tk = np.zeros((nSam, nSam))
        for i in range(nFea):
            tmk = K[i]
            tmp = w[i] * tmk
            tk = tk + tmp
        
        tmp = np.dot(np.transpose(A), tk)
        tmp = np.dot(tmp, A)        # c * c
        tmp = np.diag(tmp)
        tmp = repVec(tmp, nSam)     # c * nSam
        
        tmq = np.dot(np.transpose(A), tk)    # c * nSam
        tm = tmp - 2 * tmq      # c * nSam
        B, Index = iMin(tm, axis=0)
        
        old_Z = Z
        Z = np.zeros((nSam, c))
        for i in range(nSam):
            tid = int(Index[i])
            Z[i, tid] = 1
            #Z[:, Index] = 1
        
        # +++++ Check the termination +++++
        obj = Z - old_Z
        obj = norm(obj, 2)
        str = 'The %d-th iteration: ' %ii + '%f' %obj
        print(str)
        if obj < 1e-7:
            break
        
        # +++++ Update e +++++
        e = np.zeros((nSam, nFea))
        we = np.zeros((nSam, nFea))
        for i in range(nFea):
            tmk = K[i]
            tiik = np.diag(tmk)     # nSam * 1
            tiik = repVec(tiik, c)      # nSam * c
            tiik = np.transpose(tiik)       # c * nSam
            
            tmp = np.dot(np.transpose(A), tmk)      # c * nSam
            #tmq = np.dot(np.transpose(A), tmk)
            tmq = np.dot(tmp, A)                    # c * c
            tmq = np.diag(tmq)
            tmq = repVec(tmq, nSam)     # c * nSam
            
            tm = tiik - 2 * tmp + tmq   # c * nSam
            tm = np.dot(Z, tm)      # nSam * nSam
            tm = np.diag(tm)
            e[:, i] = tm
            
            tme = w[i] * tm
            we[:, i] = tme
            
        # +++++ Update h +++++
        tmp = np.sum(we, axis=1)        # nSam * 1
        tmp = 2 * np.sqrt(tmp)
        for i in range(nSam):
            if abs(tmp[i]) < 1e-6:
                tmp[i] = tmp[i] + 1e-6
        
        tmp = repVec(tmp, nFea)
        tmp = e / tmp       # nSam * nFea
        h = np.sum(tmp, axis=0)     # nFea * 1
        
        # +++++ Update w +++++
        tme = gamma - 1
        tme = gamma / tme
        tmp = np.power(h, tme)
        tmp = np.sum(tmp)
        tmp = float(1) / tmp
        tme = float(1) / gamma
        tmp = np.power(h, tme)
        
        tme = gamma - 1
        tme = float(1) / tme
        tmq = np.power(h, tme)
        w = tmp * tmq       # nFea
        
        del tme
        
        # +++++ Update D +++++
        tmw = np.zeros((c, nSam))
        for i in range(nFea):
            tmk = K[i]
            tiik = np.diag(tmk)     # nSam * 1
            tiik = repVec(tiik, c)      # nSam * c
            tiik = np.transpose(tiik)       # c * nSam
            
            tmp = np.dot(np.transpose(A), tmk)      # c * nSam
            #tmq = np.dot(np.transpose(A), tmk)
            tmq = np.dot(tmp, A)                    # c * c
            tmq = np.diag(tmq)
            tmq = repVec(tmq, nSam)     # c * nSam
            
            tm = tiik - 2 * tmp + tmq       # c * nSam
            tm = w[i] * tm
            tmw = tmw + tm
            
            
        tmp = np.dot(Z, tmw)        # nSam * nSam
        tmd = np.diag(tmp)      # nSam * 1
        D = np.diag(tmd)
        
        
    labels = predCluster(Z)
    
    return labels, Z, A, w
    
    
    
            
            
            
            
        
    