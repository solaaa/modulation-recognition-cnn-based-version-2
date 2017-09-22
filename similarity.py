""" 
    citation: http://blog.csdn.net/lishuhuakai/article/details/53573241

    goal: 
-------------------------------------------------------------------------
    Hausdorff distance: 
    asume:
        A = {a1, a2, ..., an}, B = {b1, b2, ..., bm} 
    H(.) represents Hausdorff distance:
        H(A, B) = max{h(A, B), h(B, A)}, 
    where
    h(A, B) = max{dist(ai, bi)}

"""  
import math  
import numpy as np  
from math import sqrt  
import sys
from numba import jit

sys.setrecursionlimit(512*512) 
def euclidean_metric(pa, pb):  
    return sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)  

@jit
def one_way_hausdorff_distance(sa, sb):  
    ''' h(a, b) '''  
    distance = 0.0  
    for pa in sa: 
        shortest = 1000  
        for pb in sb:  
            dis = euclidean_metric(pa, pb)  
            if dis < shortest:  
                shortest = dis  
        if shortest > distance:  
            distance = shortest  
    return distance  
  
def hausdorff_distance(sa, sb):  
    '''
    H(A, B) = max{h(A, B), h(B, A)}, 
    '''
    sa = reconstruction(sa)
    sb = reconstruction(sb)
    dis_a = one_way_hausdorff_distance(sa, sb)  
    dis_b = one_way_hausdorff_distance(sb, sa)  
    return dis_a if dis_a > dis_b else dis_b  
  
  

    
""" 
    Frechet distance 
"""  

  
# Euclidean distance.  
def euc_dist(pt1,pt2):  
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))  
  
def _c(ca,i,j,P,Q):  
    if ca[i,j] > -1:  
        return ca[i,j]  
    elif i == 0 and j == 0:  
        ca[i,j] = euc_dist(P[0],Q[0])  
    elif i > 0 and j == 0:  
        ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))  
    elif i == 0 and j > 0:  
        ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))  
    elif i > 0 and j > 0:  
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))  
    else:  
        ca[i,j] = float("inf")  
    return ca[i,j]  
  
""" 
Computes the discrete frechet distance between two polygonal lines 
Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf 

"""  
def frechet_distance(P,Q):
    '''
    input type : 1-d numpy.array or list
    '''
    P = reconstruction(P)
    Q = reconstruction(Q)
    ca = np.ones((len(P),len(Q)))  
    ca = np.multiply(ca,-1)  
    return _c(ca,len(P)-1,len(Q)-1,P,Q)  

def reconstruction(sequence):
    '''
    to reconstruct time sequence to dot set
    e.g.
    [a,b,c] ---> [(0,a), (1,b), (2, c)]
    -----------------------------------------------
    input type : 1-d numpy.array or list
    '''
    out = []
    sequence = list(sequence)
    for i in range(len(sequence)):
        out.append((i+1, sequence[i]))
    return out

