import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
from scipy.linalg import block_diag
from scipy.sparse.linalg import gmres
import scipy.special as sps
import warnings

from Naive import get_bc_conditions, teardrop
from Naive import sympy_kernel, test_curve_weights, sympy_kernel_teardrop, zpfunc
from Naive import compute_double_layer_kernel_test, ellipse, make_panels, ellipse_normal, teardrop_normal,test_curve_weights_teardrop
from Naive import compute_double_layer_off_boundary, get_naive_potential, get_potential, get_error_teardrop_naive
from Naive import sympy_old_kernel_teardrop
n = 16
T, W, _ = sps.legendre(n).weights.T
sympy_kernel_teardrop_global = sympy_kernel_teardrop(np.pi/2)

def compute_f_true(s,target_complex, aspect):
    npoin = s.shape[0]
    curve_nodes = ellipse(s, stretch=aspect)
    curve_nodes = curve_nodes.reshape(2,-1)
    curve_normal = np.array(ellipse_normal(s, aspect, npoin))
    complex_positions = [complex(curve_nodes[0][i],curve_nodes[1][i]) for i in range(npoin)]
    complex_positions = np.array(complex_positions)
    f_true = compute_double_layer_off_boundary(complex_positions, curve_normal, target_complex, npoin)
    return f_true

def compute_f_true_teardrop(z, nz, target_complex):
    complex_positions = z
    curve_normal = nz
    npoin = z.shape[0]
    f_true = compute_double_layer_off_boundary(complex_positions, curve_normal, target_complex, npoin)
    return f_true

def get_density_true(parametrization, weights, aspect):
    npoin = parametrization.shape[0]
    complex_positions = zfunc_ellipse(parametrization, aspect)
    complex_positions = complex_positions[0]
    curve_nodes = [complex_positions.real, complex_positions.imag]
    curve_nodes = np.array(curve_nodes)

    sympy_kern = sympy_kernel(3)
    D_K = np.zeros((npoin, npoin))
    for i in range(npoin):
        D_K[i,:] = sympy_kern.kernel_evaluate(parametrization[i],parametrization)
    for i in range(npoin):
        D_K[i,i] = sympy_kern.kernel_evaluate_equal(parametrization[i])


    W_shape = np.diag(weights * np.abs(zpfunc_ellipse(parametrization, aspect))[0])
    #print("D_K", D_K.shape)
    #print("W_Shape",np.abs(zpfunc_ellipse(parametrization, aspect))[0].shape)

    D_KW = D_K @ W_shape
    LHS = 0.5*np.eye(npoin) + D_KW
    #RHS = np.loadtxt("../InitialConditions/bc_potential.np")
    test_charge = np.array([-2,2])
    RHS = get_bc_conditions([test_charge], complex_positions)
    #assert(np.max(np.abs(RHS-get_bc_conditions([test_charge], complex_positions)))<=1e-6)
   
    density = gmres(LHS, RHS)[0]

    return density

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
    denom = 2**(nsub-(level+1)) * npan
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
    #level goes from 0 to nsub. 
    #returns a type b mesh which is used to compute
    #the kernel
    #different from the paper where level goes from 1,nsub
    #note that nsub represents a type b mesh on \tau^*!!!
    #
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

def zloc_init_old(theta, T, W, nsub, level, npan):
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
    #level goes from 0 to nsub. 
    #returns a type b mesh which is used to compute
    #the kernel
    #different from the paper where level goes from 1,nsub
    #note that nsub represents a type b mesh on \tau^*!!!
    #
    denom = 2**(nsub-(level)) * npan
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


def MAinit_ellipse_cancellation(parametrization, aspect):
    sympy_kern = sympy_kernel(aspect)
    npoin = parametrization.shape[0]
    D_K = np.zeros((npoin, npoin))
    min_vals = []
    for i in range(npoin):
        D_K[i,:] = sympy_kern.kernel_evaluate(parametrization[i],parametrization)
        c = sympy_kern.cancel_evaluate(parametrization[i],parametrization)[0][0]
        c = np.abs(np.nan_to_num(c, nan=1))
        c[i] = 1
        min_vals.append(np.min(c))
    for i in range(npoin):
        D_K[i,i] = sympy_kern.kernel_evaluate_equal(parametrization[i])

    return np.min(np.array(min_vals))

def MAinit_ellipse_exact(parametrization, weights, aspect):

    sympy_kern = sympy_kernel(aspect)
    npoin = parametrization.shape[0]
    D_K = np.zeros((npoin, npoin))
    for i in range(npoin):
        for j in range(npoin):
                if i != j:
                    D_K[i,j] = sympy_kern.kernel_evaluate_exact(parametrization[i],parametrization[j])
    for i in range(npoin):
        D_K[i,i] = sympy_kern.kernel_evaluate_equal(parametrization[i])

    W_shape = np.diag(weights * np.abs(zpfunc_ellipse(parametrization, aspect))[0])

    D_KW = D_K @ W_shape

    return 2*D_KW

def MAinit_ellipse(parametrization, weights, aspect):

    sympy_kern = sympy_kernel(aspect)
    npoin = parametrization.shape[0]
    D_K = np.zeros((npoin, npoin))
    for i in range(npoin):
        D_K[i,:] = sympy_kern.kernel_evaluate(parametrization[i],parametrization)
    for i in range(npoin):
        D_K[i,i] = sympy_kern.kernel_evaluate_equal(parametrization[i])

    W_shape = np.diag(weights * np.abs(zpfunc_ellipse(parametrization, aspect))[0])

    D_KW = D_K @ W_shape

    return 2*D_KW

def MAinit_teardrop(parametrization, weights, theta):
    #sympy_kernel_teardrop_global = sympy_kernel_teardrop(theta)
    npoin = parametrization.shape[0]
    D_K = np.zeros((npoin, npoin))
    for i in range(npoin):
        D_K[i,:] = sympy_kernel_teardrop_global.kernel_evaluate(parametrization[i],parametrization)
    for i in range(npoin):
        D_K[i,i] = sympy_kernel_teardrop_global.kernel_evaluate_equal(parametrization[i])

    W_shape = np.diag(weights * np.abs(zpfunc(parametrization, theta)))

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

def Rcomp_teardrop(theta, T, W, Pbc, PWbc, nsub, npan):
    R = None
    #This I don't particularily believe. Nvm.
    #It runs nsub+1 times.
    for level in range(0,nsub+1):
        s, w = zloc_init_ellipse(T, W, nsub, level, npan)
        K = MAinit_teardrop(s, w, theta)
        #In the paper K absorbs a factor of 2, my MAinit_ellipse doesn't have that factor of 2
        MAT = np.eye(96) + K
        if level == 0:
            R = np.linalg.inv(MAT[16:80,16:80])
        MAT[16:80,16:80] = np.linalg.inv(R)
        R = PWbc.T @ np.linalg.inv(MAT) @ Pbc
    return R



def Rcomp(theta,T,W,Pbc,PWbc,nsub,npan):
    R = None
    for level in range(0, nsub):
        z,zp,zpp,nz,w,wzp = zloc_init(theta,T,W,nsub,level,npan)
        K = MAinit(z,zp,zpp,nz,w,wzp,96)
        MAT = np.eye(96) + K
        if level == 0:
            R = np.linalg.inv(MAT[16:80,16:80])
        MAT[16:80,16:80] = np.linalg.inv(R)
        R = PWbc.T @ np.linalg.inv(MAT) @ Pbc
    return R

def f(s, target):
    return (-1/(2*np.pi)) * np.log(np.linalg.norm(s-target, axis=1))

#def f(s, target, aspect):
#    sympy_kern = sympy_kernel(aspect)
#    return sympy_kern.kernel_evaluate(s,target)



def get_param_T(end1, end2):
    return (T+1)/2 * (end2-end1) + end1



def get_teardrop_boundaries(nsub, npan):
    denom = 2 * npan

    #the first 2 panels closest to the singularity correspond
    #to tau^*
    start = 2/npan
    arr_endpoints = []
    for i in range(nsub+2):
        arr_endpoints.append(start)
        start /= 2
    arr_endpoints.append(0)

    parametrization = np.array([])
    param_weights = np.array([])
    for i in range(len(arr_endpoints)-1, 0, -1):
        #print(arr_endpoints[i],arr_endpoints[i-1])
        parametrization = np.append(parametrization ,get_param_T(arr_endpoints[i],arr_endpoints[i-1]))
        param_weights = np.append(param_weights, W*(arr_endpoints[i-1]-arr_endpoints[i]))



    start_ind = len(parametrization)
    otherpanels = np.linspace(2*(1/npan),1-2*(1/npan), npan-3)

    otherpanelsT = np.array([])
    otherweights = np.array([])
    for i in range(len(otherpanels)-1):
        otherpanelsT = np.append(otherpanelsT,get_param_T(otherpanels[i],otherpanels[i+1]))
        otherweights = np.append(otherweights, W*(otherpanels[i+1]-otherpanels[i]))

    end_in = start_ind + len(otherweights)

    parametrization = np.append(np.append(parametrization, otherpanelsT), list(reversed(1-parametrization)))
    weights = np.append(np.append(param_weights, otherweights), list(reversed(param_weights)))

    kcirc_indices = np.arange(start_ind, end_in)

    return np.append(np.append(list(reversed(arr_endpoints)), otherpanels), (1-arr_endpoints))



def give_fine_mesh_parametrization_ellipse(nsub, npan):
    denom = 2 * npan

    #the first 2 panels closest to the singularity correspond
    #to tau^*
    start = 2/npan
    arr_endpoints = []
    for i in range(nsub+2):
        arr_endpoints.append(start)
        start /= 2
    arr_endpoints.append(0)

    parametrization = np.array([])
    param_weights = np.array([])
    for i in range(len(arr_endpoints)-1, 0, -1):
        #print(arr_endpoints[i],arr_endpoints[i-1])
        parametrization = np.append(parametrization ,get_param_T(arr_endpoints[i],arr_endpoints[i-1]))
        param_weights = np.append(param_weights, W*(arr_endpoints[i-1]-arr_endpoints[i]))



    start_ind = len(parametrization)
    otherpanels = np.linspace(2*(1/npan),1-2*(1/npan), npan-3)

    otherpanelsT = np.array([])
    otherweights = np.array([])
    for i in range(len(otherpanels)-1):
        otherpanelsT = np.append(otherpanelsT,get_param_T(otherpanels[i],otherpanels[i+1]))
        otherweights = np.append(otherweights, W*(otherpanels[i+1]-otherpanels[i]))

    end_in = start_ind + len(otherweights)

    parametrization = np.append(np.append(parametrization, otherpanelsT), list(reversed(1-parametrization)))
    weights = np.append(np.append(param_weights, otherweights), list(reversed(param_weights)))

    kcirc_indices = np.arange(start_ind, end_in)
   


    return parametrization, weights/2, kcirc_indices

def get_K_star_circ_fine_teardrop(nsub, npan, theta):
    param, weights, kcirc = give_fine_mesh_parametrization_ellipse(nsub, npan)
    
    Kstar = MAinit_teardrop(param, weights, theta)
    K = MAinit_teardrop(param, weights, theta)
    kcirc = set(kcirc)

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if (i in kcirc) or (j in kcirc):
                Kstar[i,j] = 0

    return Kstar, (K-Kstar)

def get_K_star_circ_fine(nsub, npan, aspect):
    param, weights, kcirc = give_fine_mesh_parametrization_ellipse(nsub, npan)
    
    Kstar = MAinit_ellipse(param, weights, aspect)
    K = MAinit_ellipse(param, weights, aspect)
    kcirc = set(kcirc)

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if (i in kcirc) or (j in kcirc):
                Kstar[i,j] = 0

    return Kstar, (K-Kstar)

def get_K_fine(nsub, npan, aspect):
    param, weights, kcirc = give_fine_mesh_parametrization_ellipse(nsub, npan)
    K = MAinit_ellipse(param, weights, aspect)
    return K


def get_K_circ_coarse(npan, aspect):
    s, w = zinit_ellipse(T, W, npan)
    npoin = s.shape[0]

    #In the paper K absorbs a factor of 2, my MAinit_ellipse doesn't have that factor of 2
    Kcirc = MAinit_ellipse(s, w, aspect)

    starind = [i for i in range(npoin-32,npoin)] #last two panels(16 points/per panel)
    starind += [i for i in range(32)] #first two panels (16 points/per panel)
    bmask = np.zeros((Kcirc.shape[0],Kcirc.shape[1]),dtype='bool')

    for i in starind:
        for j in starind:
            bmask[i,j]=1
    Kcirc[bmask] = 0

    return Kcirc



def get_P_helper(num_blocks):
    IP, IPW = IPinit(T,  W)
    
    retMe = np.zeros(((num_blocks+2) * 16, num_blocks*16))
    retMe[16:-16,:] = np.eye(num_blocks*16)
    retMe[:32,:16] = IP
    retMe[-32:,-16:] = IP

    return retMe

def get_P(npan, nsub):
    P = get_P_helper(npan)
    for i in range(1, nsub):
        P = get_P_helper(npan+2*i) @ P
    return P
    #Note that IP takes us from a single panel to a double panel

def get_PW(npan, nsub):
    W_fin = give_fine_mesh_parametrization_ellipse(nsub, npan)[1]
    W_coarse = zinit_ellipse(T,  W, npan)[1]
    W_fin = np.diag(W_fin)
    W_coarse_inv = np.diag(1/W_coarse)
    P = get_P(npan, nsub)

    """ print("W_fin shape:", W_fin.shape)
    print("P shape:", P.shape)
    print("W_coarse shape:", W_coarse_inv.shape)  """

    PW = W_fin @ P @ W_coarse_inv

    return PW

def get_R_true_teardrop(npan, nsub, theta):
    P = get_P(npan, nsub)
    PW_T = get_PW(npan, nsub).T

    Kstar_fine = get_K_star_circ_fine_teardrop(nsub, npan, theta)[0]

    npoin = Kstar_fine.shape[0]

    R = PW_T @ np.linalg.inv(np.eye(npoin) +  Kstar_fine) @ P
    return R

def get_R_true(npan, nsub, aspect):
    P = get_P(npan, nsub)
    PW_T = get_PW(npan, nsub).T

    Kstar_fine = get_K_star_circ_fine(nsub, npan, aspect)[0]



    """ print("P shape:", P.shape)
    print("PW_T shape:", PW_T.shape)
    print("Kstar_fine shape:", Kstar_fine.shape) """
    npoin = Kstar_fine.shape[0]

    """ plt.title("PW_T")
    plt.imshow(PW_T)
    plt.show()

    plt.title("P")
    plt.imshow(P)
    plt.show()

    plt.title("K*")
    plt.imshow(Kstar_fine)
    plt.show()

    plt.title("inv(I+K*)")
    plt.imshow(np.linalg.inv(np.eye(npoin) +  Kstar_fine))
    plt.show() """
    R = PW_T @ np.linalg.inv(np.eye(npoin) +  Kstar_fine) @ P
    return R
    #Complete

def get_error_ellipse_rcip_accurate(npan, nsub):
    IP, IPW = IPinit(T,  W)

    aspect = 3

    #Number of panels = 10



    s, w = zinit_ellipse(T,  W, npan)
    z = zfunc_ellipse(s, aspect)
    z = z[0]
    npoin = s.shape[0]

    #In the paper K absorbs a factor of 2, my MAinit_ellipse doesn't have that factor of 2
    Kcirc = MAinit_ellipse_exact(s, w, aspect)

    starind = [i for i in range(npoin-32,npoin)]
    starind += [i for i in range(32)]
    bmask = np.zeros((Kcirc.shape[0],Kcirc.shape[1]),dtype='bool')

    for i in starind:
        for j in starind:
            bmask[i,j]=1
    Kcirc[bmask] = 0

    Pbc = block_diag(np.eye(16),IP,IP,np.eye(16))
    PWbc = block_diag(np.eye(16),IPW,IPW,np.eye(16))

    '''
    So one interesting thing to note is that zloc_init and zinit do 2 different things. 
    zinit should be considered the gold standard as this essentially defines
    the order in which we label nodes when constructing all of our vectors
    including our kernels. So in other words make sure that you keep this
    consistent.
    '''
    R_sp = Rcomp_ellipse_exact(aspect,T,W,Pbc,PWbc,nsub,npan)

    R = np.eye(npoin)
    #Not the most efficient but quadratic in the order of quadrature
    l=0
    for i in starind:
        m=0
        for j in starind:
            R[i,j] = R_sp[l,m]
            m+=1
        l+=1


    # get true value of R
    #R = get_R_true(npan, nsub, aspect)



    I_coa = np.eye(npoin)

    LHS = I_coa + (Kcirc@R)
    #pot_boundary = np.loadtxt('bc_potential.np')
    
    test_charge = np.array([-3,3])
    RHS = 2*get_bc_conditions([test_charge], z)

    #target = np.array([1,0.2])
    target_complex= 2 + complex(0,1)*0

    plt.figure(1)
    plt.scatter(z.real, z.imag)
    plt.scatter(test_charge[0], test_charge[1])
    plt.scatter(target_complex.real, target_complex.imag)

    #density = gmres(LHS, RHS)[0]
    density = np.linalg.solve(LHS, RHS)
    #print(LHS, RHS)
    density_hat = R @ density

    #print("LHS:", np.mean(LHS))
    #print("Kcirc:", np.mean(Kcirc))
    #print("R:", np.mean(R))

    z_list = np.empty((npoin,2))
    z_list[:,0] = z.real
    z_list[:,1] = z.imag

    f_list = compute_f_true(s, target_complex, aspect)

    awzp = w * np.abs(zpfunc_ellipse(s, aspect))

    pot_at_target = np.sum(f_list*density_hat*awzp)

    #print(pot_at_target)

    npan_naive = 11
    #out, true = get_naive_potential(npan_naive, test_charge, target_complex)
    true = get_potential(np.array([target_complex.real,target_complex.imag]), [test_charge])
    #print("RCIP Computation Error:", np.abs(pot_at_target - true))
    print(np.abs(pot_at_target - true))
    return np.abs(pot_at_target - true)

def get_error_ellipse_rcip(npan, nsub):
    IP, IPW = IPinit(T,  W)

    aspect = 3

    #Number of panels = 10



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

    '''
    So one interesting thing to note is that zloc_init and zinit do 2 different things. 
    zinit should be considered the gold standard as this essentially defines
    the order in which we label nodes when constructing all of our vectors
    including our kernels. So in other words make sure that you keep this
    consistent.
    '''
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


    # get true value of R
    #R= get_R_true(npan, nsub, aspect)
    #print(np.max(np.abs(R-R_true)))



    I_coa = np.eye(npoin)

    LHS = I_coa + (Kcirc@R)
    #pot_boundary = np.loadtxt('bc_potential.np')
    
    test_charge = np.array([-3,3])
    RHS = 2*get_bc_conditions([test_charge], z)

    #target = np.array([1,0.2])
    target_complex= 2 + complex(0,1)*0

    plt.figure(1)
    plt.scatter(z.real, z.imag)
    plt.scatter(test_charge[0], test_charge[1])
    plt.scatter(target_complex.real, target_complex.imag)

    #density = gmres(LHS, RHS)[0]
    density = np.linalg.solve(LHS, RHS)
    #print(LHS, RHS)
    density_hat = R @ density

    #print("LHS:", np.mean(LHS))
    #print("Kcirc:", np.mean(Kcirc))
    #print("R:", np.mean(R))

    z_list = np.empty((npoin,2))
    z_list[:,0] = z.real
    z_list[:,1] = z.imag

    f_list = compute_f_true(s, target_complex, aspect)

    awzp = w * np.abs(zpfunc_ellipse(s, aspect))

    pot_at_target = np.sum(f_list*density_hat*awzp)

    #print(pot_at_target)

    npan_naive = 10
    #out, true = get_naive_potential(npan_naive, test_charge, target_complex)

    true = get_potential(np.array([target_complex.real,target_complex.imag]), [test_charge])
    #print("RCIP Computation Error:", np.abs(pot_at_target - true))
    print(np.abs(pot_at_target - true))
    return np.abs(pot_at_target - true)

def main_ellipse():
    IP, IPW = IPinit(T,  W)

    aspect = 3

    #Number of panels = 10
    npan = 10
    nsub = 1

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

    '''
    So one interesting thing to note is that zloc_init and zinit do 2 different things. 
    zinit should be considered the gold standard as this essentially defines
    the order in which we label nodes when constructing all of our vectors
    including our kernels. So in other words make sure that you keep this
    consistent.
    '''
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


    # get true value of R
    #R = get_R_true(npan, nsub, aspect)



    I_coa = np.eye(npoin)

    LHS = I_coa + (Kcirc@R)
    #pot_boundary = np.loadtxt('bc_potential.np')
    
    test_charge = np.array([-2,2])
    RHS = 2*get_bc_conditions([test_charge], z)

    #target = np.array([1,0.2])
    target_complex= 1+ complex(0,1)*0.6

    density = gmres(LHS, RHS)[0]
    #print(LHS, RHS)
    density_hat = R @ density

    print("LHS:", np.mean(LHS))
    print("Kcirc:", np.mean(Kcirc))
    print("R:", np.mean(R))

    z_list = np.empty((npoin,2))
    z_list[:,0] = z.real
    z_list[:,1] = z.imag

    f_list = compute_f_true(s, target_complex, aspect)

    awzp = w * np.abs(zpfunc_ellipse(s, aspect))

    pot_at_target = np.sum(f_list*density_hat*awzp)

    #print(pot_at_target)

    npan_naive = 11
    out, true = get_naive_potential(npan_naive, test_charge, target_complex)

    print("RCIP Computation Error:", np.abs(pot_at_target - true))
    print("Naive Computation Error:", np.abs((out - true)))
    plt.scatter(z.real, z.imag)
    plt.scatter(target_complex.real, target_complex.imag)
    plt.scatter(test_charge[0], test_charge[1])
    plt.show()


def get_error_teardrop_rcip(npan, nsub):
        theta = np.pi/2

        IP, IPW = IPinit(T,  W)
        Pbc = block_diag(np.eye(16),IP,IP,np.eye(16))
        PWbc = block_diag(np.eye(16),IPW,IPW,np.eye(16))
        s, w = zinit_ellipse(T,  W, npan)
        z = zfunc(s, theta)
        npoin = s.shape[0]  

        Kcirc = MAinit_teardrop(s, w, theta)

        starind = [i for i in range(npoin-32,npoin)]
        starind += [i for i in range(32)]
        bmask = np.zeros((Kcirc.shape[0],Kcirc.shape[1]),dtype='bool')

        for i in starind:
            for j in starind:
                bmask[i,j]=1
        Kcirc[bmask] = 0
        #Experimental
        
        R_sp = Rcomp_teardrop(theta,T,W,Pbc,PWbc,nsub,npan)

        R = np.eye(npoin)
        #Not the most efficient but quadratic in the order of quadrature
        l=0
        for i in starind:
            m=0
            for j in starind:
                R[i,j] = R_sp[l,m]
                m+=1
            l+=1

        #R_true = get_R_true_teardrop(npan,nsub,theta)
        #print("\nDifference In  Norm - NSUB:", nsub, " ", np.linalg.norm(R-R_true))
        #R = R_true
        #Experimental

        I_coa = np.eye(npoin)

        LHS = I_coa + (Kcirc@R)

        test_charge = np.array([-0.25,0.4])
        RHS = 2*get_bc_conditions([test_charge], z)

        target_complex= 0.01+ complex(0,1)*0

        #density = gmres(LHS, RHS)[0]
        density = np.linalg.solve(LHS, RHS)
        #print(np.mean(LHS @ density - RHS))
        #print(LHS, RHS)
        density_hat = R @ density

        z_list = np.empty((npoin,2))
        z_list[:,0] = z.real
        z_list[:,1] = z.imag
        zp = zpfunc(s, theta)
        f_list = compute_f_true_teardrop(z, complex(0,1)*zp/np.abs(zp), target_complex)

        awzp = w * np.abs(zpfunc(s, theta))

        pot_at_target = np.sum(f_list*density_hat*awzp)

        true = get_potential(np.array([target_complex.real,target_complex.imag]), [test_charge])

        print(pot_at_target-true)
        return np.abs(pot_at_target-true)

""" def main_teardrop1():
    IP, IPW = IPinit(T,  W)

    theta = np.pi/2

    #Number of panels = 10
    npan = 70
    sinter = np.linspace(0, 1, npan+1)
    sinterdiff = np.ones(npan)/npan
    nsub = 6

    test_charge = np.array([0,0.4])
    target_complex= 0.5 + complex(0,1)*0.05
    #target = np.array([target_complex.real, target_complex.imag])

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

    R_sp = Rcomp(theta,T,W,Pbc,PWbc,nsub,npan)
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
    LHS = I_coa +1.0*(Kcirc@R)@np.abs(wzp)
    #pot_boundary = np.loadtxt('bc_potential.np')
    RHS = 2*get_bc_conditions([test_charge], z)

    density = gmres(LHS, RHS)[0]
    #print(LHS, RHS)
    density_hat = R @ density

    #print("Density:", np.mean(density))
    #print("RHS:", np.mean(RHS))
    #print("LHS:", np.mean(LHS))
    #print("KCirc:", np.mean(Kcirc))
    #print("R:", np.mean(R))

    #print(LHS)

    z_list = np.empty((npoin,2))
    z_list[:,0] = z.real
    z_list[:,1] = z.imag

    awzp = np.abs(wzp)

    f_list = compute_f_true_teardrop(z, complex(0,1)*zp/np.abs(zp), target_complex)
    pot_at_target = np.sum(f_list*density_hat*awzp)

    true = get_potential(np.array([target_complex.real,target_complex.imag]), [test_charge])
    print(pot_at_target-true) """

def Rcomp_ellipse_exact(aspect, T, W, Pbc, PWbc, nsub, npan):
    R = None
    #This I don't particularily believe. Nvm.
    #It runs nsub+1 times.
    for level in range(1,nsub+1):
        s, w = zloc_init_ellipse(T, W, nsub, level, npan)
        K = MAinit_ellipse_exact(s, w, aspect)
        #In the paper K absorbs a factor of 2, my MAinit_ellipse doesn't have that factor of 2
        MAT = np.eye(96) + K
        if level == 1:
            R = np.linalg.inv(MAT[16:80,16:80])
        MAT[16:80,16:80] = np.linalg.inv(R)
        R = PWbc.T @ np.linalg.inv(MAT) @ Pbc
    return R


def Rcomp_ellipse(aspect, T, W, Pbc, PWbc, nsub, npan):
    R = None
    #This I don't particularily believe. Nvm.
    #It runs nsub+1 times.
    for level in range(1,nsub+1):
        s, w = zloc_init_ellipse(T, W, nsub, level, npan)
        K = MAinit_ellipse(s, w, aspect)
        #In the paper K absorbs a factor of 2, my MAinit_ellipse doesn't have that factor of 2
        MAT = np.eye(96) + K
        if level == 1:
            R = np.linalg.inv(MAT[16:80,16:80])
        MAT[16:80,16:80] = np.linalg.inv(R)
        R = PWbc.T @ np.linalg.inv(MAT) @ Pbc
    return R

def Rcomp_old(theta,lamda,T,W,Pbc,PWbc,nsub,npan):
    for level in range(1, nsub+1):
        z,zp,zpp,nz,w,wzp = zloc_init_old(theta,T,W,nsub,level,npan)
        K = MAinit(z,zp,zpp,nz,w,wzp,96)
        MAT = np.eye(96) + lamda*K
        if level == 1:
            R = np.linalg.inv(MAT[16:80,16:80])
        MAT[16:80,16:80] = np.linalg.inv(R)
        R = PWbc.T @ np.linalg.inv(MAT) @ Pbc
    return R

def MAinit_old(z,zp,zpp,nz,w,wzp,npoin):
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
def old_rcip_problem(npan, nsub):
    n = 16
    T, W, _ = sps.legendre(n).weights.T

    IP, IPW = IPinit(T,  W)

    theta = np.pi/2
    lamda = 0.999
    evec = 1
    qref = 1.1300163213105365

    #Number of panels = 10

    sinter = np.linspace(0, 1, npan+1)
    sinterdiff = np.ones(npan)/npan

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

    R_sp = Rcomp_old(theta,lamda,T,W,Pbc,PWbc,nsub,npan)
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

    RHS = 2*lamda*(nz).real
    #rhotilde = gmres(LHS, RHS)[0]
    rhotilde = np.linalg.solve(LHS, RHS)
    rhohat = R @ rhotilde
    zeta = (z.real)*np.abs(wzp)
    q = np.sum(rhohat*zeta)
    error = (np.abs(qref-q)/np.abs(qref))

    print(error)

    return error



if __name__ == '__main__':
    #main_ellipse()
    #main_teardrop1()

    x = []
    array = []

    for i in range(10, 400, 10):
        print(i)
        array.append(np.log10(old_rcip_problem(i, 10)))
        x.append(np.log10(i))

    plt.scatter(x, array)
    plt.title("Python Code Directly Translated")
    plt.xlabel("npan (nsub=10)")
    plt.ylabel("rel. error")
    plt.show() 
