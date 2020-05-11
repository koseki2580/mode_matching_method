import scipy as sp
from scipy.special import sph_harm
import numpy as np
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from numpy import arctan2, sqrt
import numexpr as ne



def cart2sph(x,y,z):
    """ x, y, z :  ndarray coordinates
        ceval: backend to use: 
              - eval :  pure Numpy
              - numexpr.evaluate:  Numexpr """
              
    azimuth = ne.evaluate('arctan2(y,x)')
    xy2 = ne.evaluate('x**2 + y**2')
    elevation = ne.evaluate('arctan2(z, sqrt(xy2))')
    r = eval('sqrt(xy2 + z**2)')
    return azimuth, elevation, r

def EquiDistSphere(n):
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * (np.arange(n)+1)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)
    
    points = np.zeros((n, 3))
    points[:,0] = radius * np.cos(theta)
    points[:,1] = radius * np.sin(theta)
    points[:,2] = z
    return points

def BnMat2(N,k,r):
    B = np.zeros((N+1)**2, dtype = np.complex).reshape((N+1)**2,)
    dx = 0
    for i in range(N+1):
        A = Bn2(i,k,r)
        for j in np.arange(-i,i+1,1):
            B[dx] = A
            dx +=1
    return B

def Bn2(n,k,r):
    kr = k*r
    y = shankel(n,1,kr)
    return y

def shankel(n,k,z):
    if k == 1:
        sha = spherical_jn(n,z) + 1j*spherical_yn(n,z)
    else:
        sha = spherical_jn(n,z) - 1j*spherical_yn(n,z)
    return sha

def calcSH(N,azi,ele):
    y_n = np.array([],dtype = np.complex)
    for n in range(N+1):
        for m in np.arange(-n,n+1,1):
            y_n = np.append(y_n,sph_harm(m,n,azi,ele))
    return y_n

def calcCmn(N,k,r,pos):
    C_N = np.zeros(((N+1)**2,int(len(pos))),dtype = np.complex)
    if len(pos) == 1:
        hn = BnMat2(N,k,r)
        C_N = hn * np.conj(calcSH(N,pos[0,0],pos[0,1])).T
    else:
        hn = BnMat2(N,k,r)
        for i in range(int(len(pos[:,0]))):
            C_N[:,i] = hn * np.conj(calcSH(N,pos[i,0],pos[i,1])).T
            
    return C_N