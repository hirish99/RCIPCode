import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
from scipy.linalg import block_diag
from scipy.sparse.linalg import gmres
import scipy.special as sps


from Naive import get_bc_conditions

n = 16
T, W, _ = sps.legendre(n).weights.T

def IPinit(T, W):
    #A and AA are vandermonde matrices.
    #Essentially 1, x,  x^2, etc.
    ##For AA we have 2n points so it is a larger Vandermonde matrix.
    ##
    A = np.ones((16, 16))
    AA = np.ones((32, 16))
    #From my understanding T2 is simply now 32 points which we interpolate down to
    T2 = np.empty((T.shape[0]*2))
    T2[:T.shape[0]] = (T-1)/2
    T2[T.shape[0]:] = (T+1)/2
    
    for k in range(1, 16):
        A[:, k] = A[:, k-1] * T
        AA[:, k] =  AA[:, k-1] * T2
    IP = AA @ np.linalg.inv(A)
    
    #Construct weighted interpolation matrix
    #rmultiply, multpiplies. corresponding elements
    W_coa = np.diag(W)
    
    W2 = np.empty(W.shape[0]*2)
    W2[:W.shape[0]] = W/2
    W2[W.shape[0]:] = W/2
    W_fin = np.diag(W2)
    
    IPW = W_fin @ IP @ np.linalg.inv(W_coa)
    
    return IP, IPW

def complex_exp(theta):
    return np.cos(theta)+complex(0,1)*np.sin(theta)
#Give the parametrization of the curve from 0 to 1 here.
def zfunc(s,theta):
    return np.sin(np.pi * s) * complex_exp(theta*(s-0.5))
def zpfunc(s, theta):
    return np.pi * np.cos(np.pi * s)*complex_exp(theta*(s-0.5)) + complex(0,1)*theta*np.sin(np.pi*s)*complex_exp(theta*(s-0.5))
def zppfunc(s, theta):
    part1 = np.pi *(-np.pi)* np.sin(np.pi * s)*complex_exp(theta*(s-0.5))
    part2 = np.pi * np.cos(np.pi * s)*complex(0,1)*theta*complex_exp(theta*(s-0.5))
    part3 = complex(0,1)*theta*np.pi*np.cos(np.pi*s)*complex_exp(theta*(s-0.5))
    part4 = complex(0,1)*theta*np.sin(np.pi*s)*complex(0,1)*theta*complex_exp(theta*(s-0.5))
    
    return part1+part2+part3+part4
def zinit(theta,sinter,sinterdiff,T,W,npan):
    npoin = 16*npan #np is the number of points used for discretization in total
    s = np.zeros(npoin)
    w = np.zeros(npoin)

    for k in range(npan):
        start_in = k*16
        end_in = (k+1)*16
        #print(start_in, end_in)
        sdif = sinterdiff[k]/2
        #essentially s represents the parametrization from 0 to 1
        #We divide the values in T by 2 get a range -0.5 to 0.5
        #Then center it at the midpoint of each interval. Simple.
        s[start_in:end_in] = (sinter[k]+sinter[k+1])/2 + sdif*T
        #The weights are correspondingly scaled/2 since we are are transforming
        #from -1,1 to x,x+sinter
        w[start_in:end_in] = W*sdif
    z = zfunc(s, theta)
    zp = zpfunc(s, theta)
    zpp = zppfunc(s, theta)
    nz = -complex(0,1)*zp/np.abs(zp)
    wzp = w*zp
    
    return z, zp, zpp,nz,w,wzp, npoin

def zloc_init(theta, T, W, nsub, level, npan):
    denom = 2**(nsub-level) * npan
    s_new = np.append(np.append(T/4 + 0.25, T/4 + 0.75), T/2+1.5)/denom
    s_new = np.append(list(reversed(1-s_new)),s_new)
    w = np.append(np.append(W/4, W/4), W/2)/denom
    w = np.append(list(reversed(w)), w)
    z = zfunc(s_new, theta)
    zp = zpfunc(s_new, theta)
    zpp = zppfunc(s_new, theta)
    nz = -complex(0,1)*zp/np.abs(zp)
    wzp = w * zp
    
    return z,zp,zpp,nz,w,wzp
    
def MAinit(z,zp,zpp,nz,w,wzp,npoin):
    import warnings
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

    N = npoin
    M1 = np.zeros((N,N))
    ##Note that the formula for the Kernel is straightforward.
    for m in range(N):
        M1[:,m] = np.abs(wzp[m]) * (nz/(z[m]-z)).real

    for m in range(N):
        M1[m,m] = (-w*(zpp/zp).imag/2)[m]

    warnings.filterwarnings("default", message="divide by zero encountered in divide")
    
    retMe = (M1/np.pi)

    return retMe
    
def Rcomp(theta,lamda,T,W,Pbc,PWbc,nsub,npan):
    for level in range(1, nsub+1):
        z,zp,zpp,nz,w,wzp = zloc_init(theta,T,W,nsub,level,npan)
        K = MAinit(z,zp,zpp,nz,w,wzp,96)
        MAT = np.eye(96) + lamda*K
        if level == 1:
            R = np.linalg.inv(MAT[16:80,16:80])
        MAT[16:80,16:80] = np.linalg.inv(R)
        R = PWbc.T @ np.linalg.inv(MAT) @ Pbc
    return R

def f(s, target):
    return (-1/(2*np.pi)) * np.log(np.linalg.norm(s - target,2,axis=1))

def main():
    IP, IPW = IPinit(T,  W)

    theta = np.pi/2
    lamda = 1

    #Number of panels = 10
    npan = 10
    sinter = np.linspace(0, 1, npan+1)
    sinterdiff = np.ones(npan)/npan
    nsub = 6

    z, zp, zpp, nz, w, wzp, npoin = zinit(theta, sinter, sinterdiff, T, W, npan)

    Kcirc = MAinit(z,zp,zpp,nz,w,wzp,npoin)

    starind = [i for i in range(npoin-32,npoin)]
    starind += [i for i in range(32)]
    bmask = np.zeros((Kcirc.shape[0],Kcirc.shape[1]),dtype='bool')

    for i in starind:
        for j in starind:
            bmask[i,j]=1
    Kcirc[bmask] = 0

    Pbc = block_diag(np.eye(16),IP,IP,np.eye(16))
    PWbc = block_diag(np.eye(16),IPW,IPW,np.eye(16))

    R_sp = Rcomp(theta,lamda,T,W,Pbc,PWbc,nsub,npan)
    R = np.eye(npoin)
    #Not the most efficient but quadratic in the order of quadrature
    l=0
    for i in starind:
        m=0
        for j in starind:
            R[i,j] = R_sp[l,m]
            m+=1
        l+=1

    I_coa = np.eye(npoin)
    LHS = I_coa +lamda*(Kcirc@R)
    #pot_boundary = np.loadtxt('bc_potential.np')
    test_charge = np.array([-2,2])
    RHS = 2*get_bc_conditions([test_charge], z)

    target = np.array([-1,0.3])

    density = gmres(LHS, RHS)[0]
    #print(LHS, RHS)
    density_hat = R @ density

    z_list = np.empty((npoin,2))
    z_list[:,0] = z.real
    z_list[:,1] = z.imag

    f_list = f(z_list,target)
    awzp = np.abs(wzp)

    pot_at_target = np.sum(f_list*density_hat*awzp)

    print(pot_at_target)

if __name__ == '__main__':
    main()