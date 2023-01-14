import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
from scipy.linalg import block_diag
from scipy.sparse.linalg import gmres
import scipy.special as sps

from Naive import get_bc_conditions
from Naive import sympy_kernel, test_curve_weights
from Naive import compute_double_layer_kernel_test, ellipse, make_panels, ellipse_normal

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

    print(IPW.shape)
    
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

def zfunc_ellipse(s, a):
    return np.array([
        a*np.cos(2*np.pi*s)+
        complex(0,1)*np.sin(2*np.pi*s)
        ])

def zpfunc_ellipse(s, a):
    return np.array([
        -2*np.pi*a*np.sin(2*np.pi*s)+
        2*np.pi*complex(0,1)*np.cos(2*np.pi*s)
        ])

def zppfunc_ellipse(s, a):
    return np.array([
        -((2*np.pi)**2)*a*np.cos(2*np.pi*s)+
        -((2*np.pi)**2)*complex(0,1)*np.sin(2*np.pi*s)
        ])

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

def zinit_ellipse(T,W,npan):
    npoin = 16*npan #np is the number of points used for discretization in total

    #npan = 10
    sinter = np.linspace(0, 1, npan+1)
    sinterdiff = np.ones(npan)/npan

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
        """     z = zfunc_ellipse(s, a)
        zp = zpfunc_ellipse(s, a)
        zpp = zppfunc_ellipse(s, a)
        nz = -complex(0,1)*zp/np.abs(zp)
        wzp = w*zp """
    
    return s, w

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

def zloc_init_ellipse(T, W, nsub, level, npan):
    #level goes from 0 to nsub-1
    #different from the paper where level goes from 1,nsub
    #note that nsub represents a type b mesh on \tau^*!!!
    denom = 2**(nsub-level) * npan
    s_new = np.append(np.append(T/4 + 0.25, T/4 + 0.75), T/2+1.5)/denom
    s_new = np.append(list(reversed(1-s_new)),s_new)
    w = np.append(np.append(W/4, W/4), W/2)/denom
    w = np.append(list(reversed(w)), w)
    '''
    z = zfunc_ellipse(s_new, a)
    zp = zpfunc_ellipse(s_new, a)
    zpp = zppfunc_ellipse(s_new, a)
    nz = -complex(0,1)*zp/np.abs(zp)
    wzp = w * zp
    
    return z,zp,zpp,nz,w,wzp '''

    return s_new, w

def MAinit_ellipse_check(parametrization, weights, aspect):

    sympy_kern = sympy_kernel(aspect)
    npoin = parametrization.shape[0]
    D_K = np.zeros((npoin, npoin))
    for i in range(npoin):
        D_K[i,:] = sympy_kern.kernel_evaluate(parametrization[i],parametrization)
    for i in range(npoin):
        D_K[i,i] = sympy_kern.kernel_evaluate_equal(parametrization[i])

    W_shape = np.diag(weights) * np.abs(zpfunc_ellipse(parametrization, aspect))

    D_KW = D_K @ W_shape

    return D_KW, D_K, W_shape

def MAinit_ellipse(parametrization, weights, aspect):

    sympy_kern = sympy_kernel(aspect)
    npoin = parametrization.shape[0]
    D_K = np.zeros((npoin, npoin))
    for i in range(npoin):
        D_K[i,:] = sympy_kern.kernel_evaluate(parametrization[i],parametrization)
    for i in range(npoin):
        D_K[i,i] = sympy_kern.kernel_evaluate_equal(parametrization[i])

    W_shape = np.diag(weights) * np.abs(zpfunc_ellipse(parametrization, aspect))

    D_KW = D_K @ W_shape

    return 2*D_KW


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
    

def Rcomp_ellipse(aspect, T, W, Pbc, PWbc, nsub, npan):
    for level in range(1, nsub+1):
        s, w = zloc_init_ellipse(T, W, nsub, level, npan)
        K = MAinit_ellipse(s, w, aspect)
        #In the paper K absorbs a factor of 2, my MAinit_ellipse doesn't have that factor of 2
        MAT = np.eye(96) + K
        if level == 1:
            R = np.linalg.inv(MAT[16:80,16:80])
        MAT[16:80,16:80] = np.linalg.inv(R)
        R = PWbc.T @ np.linalg.inv(MAT) @ Pbc
    return R

def Rcomp(theta,lamda,T,W,Pbc,PWbc,nsub,npan):
    R = None
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



def get_param_T(end1, end2):
    return (T+1)/2 * (end2-end1) + end1


def give_fine_mesh_parametrization_ellipse(nsub, npan):
    denom = 2 * npan

    #the first 2 panels closest to the singularity correspond
    #to tau^*
    start = 2/npan
    arr_endpoints = []
    for i in range(nsub+1):
        arr_endpoints.append(start)
        start /= 2
    arr_endpoints.append(0)

    parametrization = np.array([])
    param_weights = np.array([])
    for i in range(len(arr_endpoints)-1, 0, -1):
        #print(arr_endpoints[i],arr_endpoints[i-1])
        parametrization = np.append(parametrization ,get_param_T(arr_endpoints[i],arr_endpoints[i-1]))
        param_weights = np.append(param_weights, W*(arr_endpoints[i-1]-arr_endpoints[i]))

    otherpanels = np.linspace(2*(1/npan),1-2*(1/npan), npan-3)
    otherpanelsT = np.array([])
    otherweights = np.array([])
    for i in range(len(otherpanels)-1):
        otherpanelsT = np.append(otherpanelsT,get_param_T(otherpanels[i],otherpanels[i+1]))
        otherweights = np.append(otherweights, W*(otherpanels[i+1]-otherpanels[i]))

    parametrization = np.append(np.append(parametrization, otherpanelsT), list(reversed(1-parametrization)))
    weights = np.append(np.append(param_weights, otherweights), list(reversed(param_weights)))

    return parametrization, weights

def get_K_star_fine(nsub, npan):
    param = give_fine_mesh_parametrization_ellipse(nsub, npan)
    K = MAinit_ellipse(param, )



    










def main_ellipse():
    IP, IPW = IPinit(T,  W)

    aspect = 3

    #Number of panels = 10
    npan = 10
    nsub = 3

    s, w = zinit_ellipse(T,  W, npan)
    z = zfunc_ellipse(s, aspect)
    z = z[0]
    npoin = s.shape[0]

    #In the paper K absorbs a factor of 2, my MAinit_ellipse doesn't have that factor of 2
    Kcirc = MAinit_ellipse(s, w, aspect)

    starind = [i for i in range(npoin-32,npoin)]
    starind += [i for i in range(32)]
    bmask = np.zeros((Kcirc.shape[0],Kcirc.shape[1]),dtype='bool')

    for i in starind:
        for j in starind:
            bmask[i,j]=1
    Kcirc[bmask] = 0

    Pbc = block_diag(np.eye(16),IP,IP,np.eye(16))
    PWbc = block_diag(np.eye(16),IPW,IPW,np.eye(16))

    R_sp = Rcomp_ellipse(aspect,T,W,Pbc,PWbc,nsub,npan)

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
    LHS = I_coa + (Kcirc@R)
    #pot_boundary = np.loadtxt('bc_potential.np')
    
    test_charge = np.array([-2,2])
    RHS = 2*get_bc_conditions([test_charge], z)

    target = np.array([0,0.2])

    density = gmres(LHS, RHS)[0]
    #print(LHS, RHS)
    density_hat = R @ density

    z_list = np.empty((npoin,2))
    z_list[:,0] = z.real
    z_list[:,1] = z.imag

    f_list = f(z_list,target)

    awzp = w * np.abs(zpfunc_ellipse(s, aspect))

    pot_at_target = np.sum(f_list*density_hat*awzp)

    print(pot_at_target)


""" 
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
    print(z)
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

    print(pot_at_target) """




if __name__ == '__main__':
    main_ellipse()