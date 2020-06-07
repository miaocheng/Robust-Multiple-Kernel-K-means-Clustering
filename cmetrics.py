# -----------------------------------------------------------------------------------------------------------
# This python file contains several essential implementations of measure functions for clustering
# 
# Reference:
# C. Zhang, H. Fu, S. Liu, G. Liu, X. Cao, Low-Rank Tensor Constrained Multiview Subspace Clustering, IEEE
# Conference on ICCV, 2015.
# Coded by Miao Cheng
# Date: 2020-03-05
# -----------------------------------------------------------------------------------------------------------
import numpy as np
from scipy import special as sspec

from cala import *


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This function calculates the similarities measure of obtained clustering labels
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cmeasure(Label, labels):
    # +++++ Normalized Mutual Information +++++
    print('The Normalized Mutual Information:')
    A, nmi, avg = calNMI(Label, labels)
    str = 'The obtained NMI: %f' %nmi + '\navg: %f' %avg + '\n'
    print(str)
    
    # +++++ Accuracy +++++
    print('The Accuracy Measure:')
    accuracy, acc = Accuracy(Label, labels)
    str = 'The obtained Accuarcy: %f' %accuracy + '\n'
    print(str)
    
    # +++++ f +++++
    print('The F Measure:')
    f, p, r = calF(Label, labels)
    str = 'The obtained f: %f' %f
    print(str)
    str = 'The obtained p: %f' %p
    print(str)
    str = 'The obtained r: %f\n' %r
    print(str)
    
    
    # +++++ RandIndex +++++
    print('The RandIndex Measure:')
    ar, ri, MI, HI = RandIndex(Label, labels)
    str = 'The obtained ar: %f' %ar
    print(str)
    str = 'The obtained ri: %f' %ri
    print(str) 
    str = 'The obtained MI: %f' %MI
    print(str)
    str = 'The obtained HI: %f\n' %HI
    print(str)
    
    
    return nmi, avg, acc, f, p, r, ar, ri, MI, HI


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# HMFLIP filip assignment state of all zeros along a path.
# 
# %[A,C,U]=hmflip(A,C,LC,LR,U,l,r)
# Input:
# A   - the cost matrix.
# C   - the assignment vector.
# LC  - the column label vector.
# LR  - the row label vector.
# U   - the 
# r,l - position of last zero in path.
# Output:
# A   - updated cost matrix.
# C   - updated assignment vector.
# U   - updated unassigned row list vector.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hmflip(A, C, LC, LR, U, l, r):
    nRow, nCol = np.shape(A)
    
    while True:
        # +++++ Move assignment in column l to row r +++++
        C[l] = r + 1
        
        # +++++ Find zero before this +++++
        tmp = A[r, :]
        m = seArr(-(l + 1), tmp)
        
        # +++++ Link past this zero +++++
        if m != []:
            tmp = A[r, l]
            A[r, m] = tmp
        
        A[r, l] = 0
        
        # +++++ If this was the first zero of the path +++++
        if LR[r] < 0:
            #U[nRow+1] = U[r]
            U[nRow] = U[r]
            U[r] = 0
            
            return A, C, U
        else:
            # +++++ Move back in this row along the path and get column of next zero +++++
            l = LR[r]
            l = int(l - 1)
            
            # +++++ Insert zero at (r, l) first in zero list +++++
            #tmp = A[r, nRow+1]
            tmp = A[r, nRow]
            A[r, l] = tmp
            
            #A[r, nRow+1] = -(l + 1)
            A[r, nRow] = -(l + 1)
            
            # +++++ Continue back along the column to get row of next zero in path +++++
            r = LC[l]
            r = int(r - 1)
            
    return A, C, U 


def findZero(L, isZero):
    nLen = len(L)
    
    ind = []
    if isZero == True:
        for i in range(nLen):
            if L[i] == 0:
                ind.append(i)
    else:
        for i in range(nLen):
            if L[i] != 0:
                ind.append(i)
                
    return ind


def reZero(L, isZero):
    nLen = len(L)
    
    tl = np.zeros((nLen, 1)).reshape(nLen, )
    if isZero == True:
        for i in range(nLen):
            if L[i] == 0:
                tl[i] = 1
                
    else:
        for i in range(nLen):
            if L[i] != 0:
                tl[i] = 1
        
    return tl


def getMinimum(M, idx, idy):
    nRow, nCol = np.shape(M)
    
    mVal = 1e7
    for i in idx:
        for j in idy:
            tmp = M[i, j]
            
            if tmp < mVal:
                mVal = tmp
                ix = i
                iy = j
                
    return mVal, ix, iy
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This function reduces parts of cost matrix in the Hungerian method. 
# 
# [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
# Input:
# A   - Cost matrix.
# CH  - vector of column of 'next zeros' in each row.
# RH  - vector with list of unexplored rows.
# LC  - column labels.
# RC  - row labels.
# SLC - set of column labels.
# SLR - set of row labels.
#
# Output:
# A   - Reduced cost matrix.
# CH  - Updated vector of 'next zeros' in each row.
# RH  - Updated vector of unexplored rows.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hmreduce(A, CH, RH, LC, LR, SLC, SLR):
    nRow, nCol = np.shape(A)
    
    # +++++ Find which rows are covered, i.e., unlabeled +++++
    coveredRows = reZero(LR, True)
    
    # +++++ Find which columns are covered, i.e., labeled +++++
    coveredCols = reZero(LC, False)
    
    r = findZero(coveredRows, True)
    c = findZero(coveredCols, True)
    
    # +++++ Get Minimum of uncovered Elements +++++
    m, ix, iy = getMinimum(A, r, c)
    
    # +++++ Subtract minimum from all uncovered elements +++++
    #A[r, c] = A[r, c] - m
    for ii in r:
        for jj in c:
            A[ii, jj] = A[ii, jj] - m
    
    #SLR = SLR[0]
    # +++++ Check all uncovered columns +++++
    for j in c:
        # +++++ and uncovered rows in path order +++++
        for i in SLR:
            i = int(i - 1)
            # +++++ If this is a (new) zero +++++
            if A[i, j] == 0:
                # +++++ If the row is not in unexplored list +++++
                if RH[i] == 0:
                    # +++++ Insert it first in unexplored list +++++
                    #RH[i] = RH[nRow+1]
                    #RH[nRow+1] = i
                    RH[i] = RH[nRow]
                    RH[nRow] = i + 1                    
                    
                    # +++++ Mark this zero as "next free" in this row +++++
                    CH[i] = j + 1
                    
                # +++++ Find last unassigned zero on row I +++++
                row = A[i, :]
                ind = []
                for ii in range(nCol):
                    if row[ii] < 0:
                        ind.append(ii)
                
                #ind = ind[0]
                #ind = imv(ind)        
                tmp = row[ind]
                colsInList = - tmp
                
                l = []
                if ( (list(colsInList) == []) | (len(colsInList) == 0) ):
                    # +++++ No zeros in the list +++++
                    tmp = nRow + 1
                    l.append(tmp)
                else:
                    tmn = len(colsInList)
                    #tmp = row[colsInList]
                    
                    for jj in range(tmn):
                        tid = colsInList[jj]
                        tid = int(tid - 1)
                        if row[tid] == 0:
                            l.append(tid + 1)
                
                if len(l) == 1:
                    tl = l[0] - 1
                else:
                    tl = l - 1
                
                A[i, tl] = - (j + 1)
                
    # +++++ Add minimum to all doubly covered elements +++++
    r = findZero(coveredRows, False)
    c = findZero(coveredCols, False)
    
    tRow = len(r)
    tCol = len(c)
    tm = np.zeros((tRow, tCol))
    for i in range(tRow):
        for j in range(tCol):
            tir = r[i]
            tic = c[j]
            tm[i, j] = A[tir, tic]
    
    # +++++ Take care of the zeros we will remove +++++
    idx = []
    idy = []
    for i in range(tRow):
        for j in range(tCol):
            if tm[i, j] <= 0:
                idx.append(i)
                idy.append(j)
    
    
    #idx = np.array(idx)
    #idy = np.array(idy)
    #idx = imv(idx)
    #idy = imv(idy)
    r = np.array(r)
    c = np.array(c)
    ix = r[idx]
    iy = c[idy]
    
    iLen = len(ix)
    for i in range(iLen):
        # +++++ Find zero before this in this row +++++
        tid = ix[i]
        tjd = iy[i]
        tma = A[tid, :]
        tmp = - (tjd + 1)
        lj = seArr(tmp, tma)
        
        # +++++ Link past it +++++
        A[tid, lj] = A[tid, tjd]
        
        # +++++ Mark it as assigned +++++
        A[tid, tjd] = 0
        
        
    A[ix, iy] = A[ix, iy] + m
    
    
    return A, CH, RH


def getSub(c, val, isEqual):
    nLen = len(c)
    
    ind = []
    if isEqual == True:
        for i in range(nLen):
            if c[i] == val:
                ind.append(i)
                
    else:
        for i in range(nLen):
            if c[i] != val:
                ind.append(i)
            
    return ind


def hminiass(A):
    nRow, nCol = np.shape(A)
    
    c = np.zeros((1, nRow)).reshape(nRow, )
    u = np.zeros((1, nRow+1))
    
    # +++++ Initialize last / next zero "pointers" +++++
    lz = np.zeros((1, nRow)).reshape(nRow, )
    nz = np.zeros((1, nRow)).reshape(nRow, )
    
    for i in range(nRow):
        # +++++ Set j to first unassigned zero in row i +++++
        #lj = nRow + 1
        lj = nRow
        
        j = - A[i, lj]
        
        # +++++ Repeat until we have no more zeros (j == 0) or we find a zero 
        # in an unassigned column (c(j) == 0) +++++
        j = int(j-1)
        
        if j == 0:
            j = int(0)
        
        while (c[j] != 0):
            lj = int(j)
            j = - A[i, lj]
            
            j = int(j - 1)
            
            if j == -1:
                break
        
        
        if j != -1:
            if j == 0:
                j = 0
            
            # +++++ Assign row i to column j +++++
            c[j] = i + 1
            
            # +++++ Remove A[i, j] from unassigned zero list +++++
            A[i, lj] = A[i, j]
            
            # +++++ Update next / last unassigned zero pointers +++++
            nz[i] = - A[i, j]
            lz[i] = lj
            
            # +++++ Indicate A[i, j] is an assigned zero +++++
            A[i, j] = 0
                             
        else:
            # +++++ Check all zeros in this row +++++
            #lj = nRow + 1
            lj = nRow
            
            j = - A[i, lj]
            j = int(j-1)
            
            while j != -1:
                if j == 0:
                    j = 0                
                
                r = int(c[j])
                r = int(r-1)
                
                # +++++ Pick up the last / next pointers +++++
                lm = int(lz[r])
                m = int(nz[r])
                lm = lm - 1
                m = m - 1
                
                # +++++ Check all unchecked zeros in free list of this row +++++
                while m != -1:
                    if c[m] == 0:
                        break
                    
                    lm = m
                    m = - A[r, lm]
                    
                if m == -1:
                    lj = j
                    j = - A[i, lj]
                    j = int(j - 1)
                    
                else:
                    A[r, lm] = - (j + 1)
                    A[r, j] = A[r, m]
                    
                    nz[r] = - A[r, m]
                    lz[r] = j + 1
                    
                    A[r, m] = 0
                    
                    c[m] = r + 1
                    
                    # +++++ Remove A[i, j] from unassigned list +++++
                    A[i, lj] = A[i, j]
                    
                    # +++++ Update last / next pointers in row r +++++
                    nz[i] = - A[i, j]
                    lz[i] = lj + 1
                    
                    # +++++ Mark A[r, m] as an assigned zero in the matrix +++++
                    A[i, j] = 0
                    
                    # +++++ and in the assignment vector +++++
                    c[j] = i + 1
                    
                    break
                
    r = np.zeros((1, nRow)).reshape(nRow, )
    ind = getSub(c, 0, False)
    ind = imv(np.array(ind))
                      
    rows = c[ind]
    rows = imv(rows)
    
    ind = np.array(rows) - 1
    r[ind] = rows
    ide = seArr(0, r)
    
    # +++++ Create vector with linked list of unassigned rows +++++
    U = np.zeros((1, nRow+1)).reshape(nRow+1, )
    #tmp = (nRow + 1) * np.ones((1, 1))
    tmp = nRow * np.ones((1, 1))
    
    empty = np.reshape(ide, (1, len(ide)))
    
    tid = np.column_stack((tmp, empty))
    tmp = np.zeros((1, 1))
    tjd = np.column_stack((empty + 1, tmp))
    
    # +++++ Translate tid to standard indices +++++
    tid = tid[0]
    tid = imv(tid)
    tjd = tjd[0]   
    
    U[tid] = tjd
    
    return A, c, U
    
    
def hminired(A):
    nRow, nCol = np.shape(A)  
    
    # +++++ Subtract column-minimum values from each column +++++
    colMin, index = iMin(A, axis=0)
    colMin = np.array(colMin)
    tmp = repVec(colMin, nCol)
    tmp = np.transpose(tmp)
    A = A - tmp
    
    # +++++ Subtract row-minimum values from each row +++++
    rowMin, indey = iMin(np.transpose(A), axis=0)
    #rowMin = np.transpose(rowMin)
    tmp = repVec(rowMin, nCol)
    A = A - tmp    
    
    # +++++ Get positions of all zeros +++++
    rInd, cInd = findData(0, A)
    cInd = np.array(cInd)
    
    # +++++ Extend A to give room for row zero list header column +++++
    z = np.zeros((nRow, 1))
    A = np.column_stack((A, z))
    for i in range(nCol):
        ind = seArr(i, rInd)
        ind = np.array(ind)
        ind = imv(ind)
        cols = cInd[ind]
        
        cols = np.reshape(cols, (1, len(cols)))
        # +++++ Insert pointers in matrix +++++
        #tmd = (nCol+1) * np.ones((1, 1))
        tmd = nCol * np.ones((1, 1))
        
        tmd = np.column_stack((tmd, cols))
        tnd = np.zeros((1, 1))
        tnd = np.column_stack((- (cols+1), tnd))
        
        # +++++ Translate tmd to suitable indices +++++
        tmd = tmd[0]
        tmd = imv(tmd)
        
        A[i, tmd] = tnd
        
        
    return A
    
    
def hminired(A):
    nRow, nCol = np.shape(A)  
    
    # +++++ Subtract column-minimum values from each column +++++
    colMin, index = iMin(A, axis=0)
    colMin = np.array(colMin)
    tmp = repVec(colMin, nCol)
    tmp = np.transpose(tmp)
    A = A - tmp
    
    # +++++ Subtract row-minimum values from each row +++++
    rowMin, indey = iMin(np.transpose(A), axis=0)
    #rowMin = np.transpose(rowMin)
    tmp = repVec(rowMin, nCol)
    A = A - tmp    
    
    # +++++ Get positions of all zeros +++++
    rInd, cInd = findData(0, A)
    cInd = np.array(cInd)
    
    # +++++ Extend A to give room for row zero list header column +++++
    z = np.zeros((nRow, 1))
    A = np.column_stack((A, z))
    for i in range(nCol):
        ind = seArr(i, rInd)
        ind = np.array(ind)
        ind = imv(ind)
        cols = cInd[ind]
        
        cols = np.reshape(cols, (1, len(cols)))
        # +++++ Insert pointers in matrix +++++
        #tmd = (nCol+1) * np.ones((1, 1))
        tmd = nCol * np.ones((1, 1))
        
        tmd = np.column_stack((tmd, cols))
        tnd = np.zeros((1, 1))
        tnd = np.column_stack((- (cols+1), tnd))
        
        # +++++ Translate tmd to suitable indices +++++
        tmd = tmd[0]
        tmd = imv(tmd)
        
        A[i, tmd] = tnd
        
        
    return A
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Hungarian solves the assignment problem using the Hungarian method.
# [C, T] = hungarian(A)
# 
# A - a square cost matrix
# C - the optimal assignment
# T - the cost of the optimal assignment
# s.t., T = trace(A(C, :)) is minimized over all possible assignments.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hungarian(A):
    nRow, nCol = np.shape(A)
    assert nRow == nCol, 'The length of row and column are not identical !'
    
    orig = A
    
    # +++++ Reduce matrix +++++
    A = hminired(A)
    
    # +++++ Do an initial assignment +++++
    A, C, U = hminiass(A)
    
    # +++++ Repeat while we have unassigned rows +++++
    #while U[nCol+1]:
    while U[nCol]:
        LR = np.zeros((1, nCol)).reshape(nCol, )
        LC = np.zeros((1, nCol)).reshape(nCol, )
        CH = np.zeros((1, nCol)).reshape(nCol, )
        RH = np.zeros((1, nCol))
        
        tmp = -1 * np.ones((1, 1))
        RH = np.column_stack((RH, tmp))
        RH = np.reshape(RH, (nCol+1, ) )
        
        SLC = []
        # +++++ Start path in the first unassigned row +++++
        r = U[nCol]
        
        # +++++ Mark row with end-of-path label +++++
        r = int(r-1)
        LR[r] = -1
        
        # +++++ Insert row first in labeled row set +++++
        SLR = []
        SLR.append(r+1)
        
        # +++++ Repeat until we manage to find an assignable zero +++++
        while (True):
            # +++++ If there are free zeros in row r +++++
            
            if A[r, nCol] != 0:
                l = - A[r, nCol]
                l = int(l-1)
                
                # +++++ If there are more free zeros in row r and row r in not 
                # yet marked as unexpected +++++
                if (A[r, l] != 0) & (RH[r] == 0):
                    RH[r] = RH[nCol]
                    RH[nCol] = r
                    
                    CH[r] = - A[r, l]
                    
            else:
                # +++++ If all rows are explored +++++
                if RH[nCol] <= 0:
                    A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)
                
                # +++++ Re-start with first unexplored row +++++    
                r = RH[nCol]
                r = int(r-1)
                
                # +++++ Get column of next free zero in row r +++++
                l = CH[r]
                
                # +++++ Advance "column of next free zero" +++++
                l = int(l - 1)
                CH[r] = - A[r, l]
                
                # +++++ If this zero is last in the list +++++
                if A[r, l] == 0:
                    RH[nCol] = RH[r]
                    RH[r] = 0
                
                
            # +++++ While the column l is labelled, i.e., in path +++++
            while LC[l] != 0:
                if RH[r] == 0:
                    if RH[nCol] <= 0:
                        A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)
                        
                    r = RH[nCol]
                    r = int(r - 1)
                    
                # +++++ Get column of next free zero in row r +++++
                l = CH[r]
                
                # +++++ Advance "column of next free zero" +++++
                l = int(l - 1)
                CH[r] = - A[r, l]
                
                # +++++ If this zero is last in list +++++
                if A[r, l] == 0:
                    RH[nCol] = RH[r]
                    RH[r] = 0
                    
                    
            # +++++ If the column found is unassigned +++++
            if C[l] == 0:
                A, C, U = hmflip(A, C, LC, LR, U, l, r)
                break
            
            else:
                # +++++ Label column l with row r +++++
                LC[l] = r + 1
                
                # +++++ Add l to the set of labeled columns +++++
                tid = l + 1
                tmp = tid * np.ones((1, 1))
                if SLC == []:
                    SLC.append(tid)
                else:
                    SLC = np.column_stack((SLC, tmp))
                
                # +++++ Continue with the row assigned to column l +++++
                r = C[l]
                r = int(r-1)
                
                # +++++ Label row r with column l +++++
                LR[r] = l + 1
                
                # +++++ Add r to the set of labeled rows +++++
                SLR.append(r + 1)
                
    # +++++ Calculate the total cost +++++
    #tmp = c[0:nCol]
    #ind = getSub(tmp, 0, False)
    T = 0
    for i in range(nCol):
        tid = int(C[i] - 1)
        tmp = orig[tid, i]
        if tmp != 0:
            T = T + tmp
        
    return C, T
    
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This function permutes labels of L2 to match with L1 as possible
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def bestMap(L1, L2):
    nLen1 = len(L1)
    nLen2 = len(L2)
    
    if nLen1 != nLen2:
        print('The length of L1 and L2 are not identical !')
        
    uL1 = np.unique(L1)
    uL2 = np.unique(L2)
    nL1 = len(uL1)
    nL2 = len(uL2)
    
    n = max(nL1, nL2)
    G = np.zeros((n, n))
    for i in range(nL1):
        tid = uL1[i]
        idx = seArr(tid, L1)
        for j in range(nL2):
            tjd = uL2[j]
            idy = seArr(tjd, L2)
            
            index, s = matind(idx, idy)
            G[i, j] = s
            
    tmp = - G
    c, t = hungarian(tmp)
    
    newL2 = np.zeros((nLen2, 1)).reshape(nLen2, )
    for i in range(nL2):
        tmp = uL2[i]
        ind = seArr(tmp, L2)
        
        tid = int(c[i] - 1)
        tmq = uL1[tid]
        newL2[ind] = tmq
        
        
    return newL2
    
    
def matind(ind, inds):
    m = len(ind)
    n = len(inds)
    
    index = []
    s = 0
    for i in range(m):
        tid = ind[i]
        for j in range(n):
            tjd = inds[j]
            if tid == tjd:
                index.append(tid)
                s = s + 1
                
    return index, s
    

def calNMI(Label, labels):
    nLen = len(Label)
    uL = np.unique(Label)
    uLs = np.unique(labels)
    nL = len(uL)
    nLs = len(uLs)
    
    # +++++ Calculate number of points in each class +++++
    D = np.zeros((nL, 1)).reshape(nL, )
    for i in range(nL):
        tml = uL[i]
        ind = seArr(tml, Label)
        tmp = np.zeros((nLen, 1)).reshape(nLen, )
        tmp[ind] = 1
        D[i] = np.sum(tmp)
        
    # +++++ Calculate Mutual Information +++++
    mi = 0
    A = np.zeros((nLs, nL))
    miarr = np.zeros((nLs, nL))
    avg = 0
    
    B = np.zeros((nLs, 1)).reshape(nLs, )
    for i in range(nLs):
        tml = uLs[i]
        inds = seArr(tml, labels)
        tmp = np.zeros((nLen, 1)).reshape(nLen, )
        tmp[inds] = 1
        B[i] = np.sum(tmp)
        
        for j in range(nL):
            tml = uL[j]
            ind = seArr(tml, Label)
            tind, s = matind(ind, inds)
            A[i, j] = s
            
            if A[i, j] != 0:
                tmp = A[i, j] / nLen
                tmq = B[i] * D[j]
                tmq = (nLen * A[i, j]) / tmq
                tmq = np.log2(tmq)
                
                miarr[i, j] = tmp * tmq
                tmp = B[i] / nLen
                tmq = A[i, j] / B[i]
                tmr = A[i, j] / B[i]
                tmr = np.log2(tmr)
                tm = tmp * tmq * tmr
                avg = avg - tm
                
            else:
                miarr[i, j] = 0
                
            mi = mi + miarr[i, j]
            
            
    # +++++ Class Entropy +++++
    class_ent = 0
    for i in range(nL):
        tmp = D[i] / nLen
        tmq = nLen / D[i]
        tmq = np.log2(tmq)
        tm = tmp * tmq
        
        class_ent = class_ent + tm
        
    # +++++ Clustering Entropy +++++
    clust_ent = 0
    for i in range(nLs):
        tmp = B[i] / nLen
        
        if B[i] < 1e-6:
            B[i] = B[i] + 1e-6
            
        tmq = nLen / B[i]
        tmq = np.log2(tmq)
        tm = tmp * tmq
        
        clust_ent = clust_ent + tm
        
        
    nmi = 2 * mi  / (clust_ent + class_ent)
    
    
    return A, nmi, avg


def Accuracy(Label, labels):
    C = bestMap(Label, labels)
    
    mLen = len(Label)
    nLen = len(C)
    assert mLen == nLen, 'The length of two label sets are not identical !'
    
    acc = 0
    for i in range(mLen):
        if Label[i] == C[i]:
            acc = acc + 1
            
    accuracy = float(acc) / mLen
    
    return accuracy, acc


def calF(Label, labels):
    mLen = len(Label)
    nLen = len(labels)
    assert mLen == nLen, 'The length of two label sets are not identical !'
    
    nL = 0
    nls = 0
    nI = 0
    for i in range(mLen):
        tml = Label[i]
        tmp = Label[i+1::]
        ind = seArr(tml, tmp)
        Ln = np.zeros((mLen-i, 1)).reshape(mLen-i, )
        Ln[ind] = 1
        
        tml = labels[i]
        tmp = labels[i+1::]
        ind = seArr(tml, tmp)
        lsn = np.zeros((mLen-i, 1)).reshape(mLen-i, )
        lsn[ind] = 1
        
        nL = nL + np.sum(Ln)
        nls = nls + np.sum(lsn)
        tmp = Ln * lsn
        nI = nI + np.sum(tmp)
        
    p = 1
    r = 1
    f = 1
    
    if nls > 0:
        p = float(nI) / nls
        
    if nL > 0:
        r = float(nI) / nL
        
    tmp = p + r
    if tmp == 0:
        f = 0
    else:
        f = 2 * p * r
        f = f / tmp
        
        
    return f, p, r


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Contingency form contigency matrix for two vectors 
# C = Contingency(v1, v2) returns contingency matrix for two column vectors
# v1, v2. These define which cluster each entity has been assigned to.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def contingency(v1, v2):
    mLen = len(v1)
    nLen = len(v2)
    assert mLen > 1, 'The length of vector 1 is incorrect !'
    assert nLen > 1, 'The length of vector 2 is incorrect !'
    
    mVal = int(np.max(v1))
    nVal = int(np.max(v2))
    
    Cont = np.zeros((mVal, nVal))
    
    for i in range(mLen):
        idx = v1[i]
        idy = v2[i]
        
        # +++++ Translate indices to suitable ones +++++
        idx = int(idx)
        idy = int(idy)
        idx = idx - 1
        idy = idy - 1
        
        Cont[idx, idy] = Cont[idx, idy] + 1
        
        
    return Cont        
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# RANDINDEX - calculates the Rand Indices to cmopare two partitions 
# ARI = RANDINDEX(c1, c2), where c1, c2 are vectors listing the 
# class membership, returns the "Hubert & Arable adjusted Rand index".
# [AR, RI, MI, HI] = RANDINDEX(c1, c2) returns the adjusted Rand index,
# the unadjusted Rand index, "Mirkin 's" index and "Hubert 's" index.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def RandIndex(Label, labels):
    C = contingency(Label, labels)
    
    n = np.sum(np.sum(C))
    tmp = np.sum(C, axis=1)
    tmp = tmp ** 2
    nis = np.sum(tmp)
    
    tmp = np.sum(C, axis=0)
    tmp = tmp ** 2
    njs = np.sum(tmp)
    
    t1 = sspec.comb(n, 2)
    tmp = C ** 2
    t2 = np.sum(np.sum(tmp))
    t3 = nis + njs
    t3 = 0.5 * t3
    
    # +++++ Expected index (for arguments) +++++
    tmp = n ** 2 + 1
    tmp = n * tmp
    tmq = (n+1) * nis
    tmr = (n+1) * njs
    tms = nis * njs
    tms = 2 * tms / n
    
    tm = tmp - tmq - tmr + tms
    tn = 2 * (n-1)
    nc = tm / tn
    
    A = t1 + t2 - t3
    D = - t2 + t3
    
    if t1 == nc:
        AR = 0
    else:
        tmp = A - nc
        tmq = t1 - nc
        AR = tmp / tmq
        
    
    RI = A / t1         # Rand 1971, Probability of agreement
    MI = D / t1         # Mirkin 1970, p(disagreement)
    tmp = A - D         
    HI = tmp / t1       # Hubert 1977, p(agree) - p(disagree)
    
    
    return AR, RI, MI, HI
    
    
        
        
        
        
        
    
            
        
        
        
