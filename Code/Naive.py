import numpy as np
import scipy.special as sps
import numpy.linalg as la
import matplotlib.pyplot as plt
import cmath as cm
from scipy.sparse.linalg import gmres
import sympy as sp
import warnings

n = 16
lege_nodes, lege_weights, _ = sps.legendre(n).weights.T

def MAinit(z,zp,zpp,nz,w,wzp,npoin):
    import warnings
    warnings.filterwarnings("ignore")

    N = npoin
    M1 = np.zeros((N,N))
    ##Note that the formula for the Kernel is straightforward.
    for m in range(N):
        M1[:,m] = np.abs(wzp[m]) * (nz/(z[m]-z)).real

    for m in range(N):
        M1[m,m] = (-w*(zpp/zp).imag/2)[m]

    
    retMe = (M1/np.pi)

    return retMe

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

class sympy_kernel:
    def __init__(self, aspect):
        import warnings
        warnings.filterwarnings("ignore")

        t_p = sp.Symbol('t_p')
        t = sp.Symbol('t')
        a = sp.Symbol('a')

        self.aspect = aspect
        self.t_p = t_p
        self.t = t
        self.a = a

        rp = sp.Matrix([a*sp.cos(2*sp.pi*t_p),sp.sin(2*sp.pi*t_p)])
        r = sp.Matrix([a*sp.cos(2*sp.pi*t),sp.sin(2*sp.pi*t)])
        vp = sp.Matrix([-sp.cos(2*sp.pi*t_p),-a*sp.sin(2*sp.pi*t_p)])/sp.sqrt((sp.cos(2*sp.pi*t_p))**2+(a*sp.sin(2*sp.pi*t_p))**2)

        numerator = (vp.T* (r-rp))[0]
        denominator = 2*sp.pi*((r-rp).T * (r-rp))[0]
        expr = numerator/denominator

        expr = expr.subs([(a, aspect)])
        f = sp.utilities.lambdify([t,t_p],expr,"numpy")
        self.kernel_lambda = f


        expr2 = (r-rp).T * (r-rp)
        expr2 = expr2.subs([(a, aspect)])
        g = sp.utilities.lambdify([t,t_p],expr2,"numpy")

        
        self.cancellation = g

    def kernel_evaluate_equal(self, t_in):
        numerator = self.aspect
        denominator = (4*np.pi)*(self.aspect**2 * (np.sin(2*np.pi*t_in))**2 + (np.cos(2*np.pi*t_in))**2)**(1.5)
        return numerator/denominator

    def kernel_evaluate(self, t_in, t_p_in):
        return self.kernel_lambda(t_in, t_p_in)
    
    def cancel_evaluate(self, t_in, t_p_in):
        return self.cancellation(t_in, t_p_in)


def G(x,y):
    return (-1/(2*np.pi))*np.log(np.linalg.norm(x-y,2))

def get_potential(point, test_charges):
    potential = 0
    for j in range(len(test_charges)):
        potential+=G(point,test_charges[j])
    return potential

def get_bc_conditions(test_charges, complex_positions):
    potential = np.zeros(len(complex_positions))
    for i in range(len(complex_positions)):
        for j in range(len(test_charges)):
            potential[i] += G(np.array([complex_positions[i].real,complex_positions[i].imag]),test_charges[j])
    
    return potential

def lege_deriv(n, x):
    if n == 0:
        return 0*x
    else:
        # https://dlmf.nist.gov/18.9#E18
        return (
            (n+1)*(2*n+2)*x*sps.eval_legendre(n, x)
            - 2*(n+1)*(n+1)*sps.eval_legendre(n+1, x)
            )/(2*n+2)/(1-x**2)

def make_deriv_matrix():
    V = np.array([
        sps.eval_legendre(i, lege_nodes)
        for i in range(n)
        ]).T
    Vprime = np.array([
        lege_deriv(i, lege_nodes)
        for i in range(n)
        ]).T
    return Vprime @ la.inv(V)

D = make_deriv_matrix()

def test_derivative():
    for i in range(n):
        f = lege_nodes**i
        if i :
            df_exact = i*lege_nodes**(i-1)
        else:
            df_exact = 0
        assert la.norm(df_exact - D@f) < 1e-12

def ellipse(t, stretch=3):
    return np.array([
        stretch*np.cos(2*np.pi*t),
        np.sin(2*np.pi*t),
        ])

def teardrop(t, theta = np.pi/2):
    complex_pos = zfunc(t, theta)
    return np.array([
        complex_pos.real, complex_pos.imag
    ])

def teardrop_normal(t, theta = np.pi/2):
    ctan = zpfunc(t, theta)
    cnormal = -complex(0,1)*ctan/np.abs(ctan)
    return np.array([
        cnormal.real, cnormal.imag
    ])

def ellipse_normal(t, stretch, npoin):
    tan = np.array([
        -stretch*2*np.pi*np.sin(2*np.pi*t),
        2*np.pi*np.cos(2*np.pi*t)
    ])
    tan = tan.reshape(2, -1)
    for i in range(tan.shape[1]):
        norm = np.linalg.norm(tan[:,i],2)
        tan[0][i] = tan[0][i]/norm
        tan[1][i] = tan[1][i]/norm
        #assert np.abs(np.linalg.norm(tan[:,i],2) - 1) <= 1e-6
  
    complex_tangent = [complex(tan[0][i],tan[1][i]) for i in range(npoin)]
    complex_normal = [complex_tangent[i]*complex(0,1) for i in range(npoin)]
    return complex_normal

def make_panels(panel_boundaries):
    panel_lengths = np.diff(panel_boundaries)
    panel_centers = 0.5*(panel_boundaries[1:] + panel_boundaries[:-1])

    return (
            panel_lengths.reshape(-1, 1) * 0.5 * lege_nodes
            + panel_centers.reshape(-1, 1))

def curve_deriv_calc(D, curve_nodes):
    return np.einsum("ij,xej->xei", D, curve_nodes)

def true_deriv(t, stretch):
    return np.array([
    -stretch*2*np.pi*np.sin(2*np.pi*t),
    2*np.pi*np.cos(2*np.pi*t)])

def fixed_curve_deriv(param, aspect):
    curve_deriv = np.empty((2, param.shape[0], param.shape[1]))
    for panel_idx in range(param.shape[0]):
        for param_idx in range(param.shape[1]):
            curve_deriv[:, panel_idx, param_idx] = true_deriv(param[panel_idx, param_idx], aspect)
    return curve_deriv

def test_curve_speed_teardrop(npan, theta):
    panel_boundaries = np.linspace(0, 1, npan+1) 
    param = make_panels(panel_boundaries)
    curve_nodes = teardrop(param, theta=theta)
    curve_deriv = curve_deriv_calc(D, curve_nodes)
    curve_speed = (curve_deriv[0]**2 + curve_deriv[1]**2)**0.5
    return curve_speed


def test_curve_speed(npan, aspect):
    panel_boundaries = np.linspace(0, 1, npan+1) 
    param = make_panels(panel_boundaries)
    curve_nodes = ellipse(param, stretch=aspect)
    curve_deriv = curve_deriv_calc(D, curve_nodes)
    #curve_deriv = fixed_curve_deriv(param, aspect)
    curve_speed = (curve_deriv[0]**2 + curve_deriv[1]**2)**0.5
    return curve_speed
    
def test_curve_weights(npan, aspect):
    #print(lege_weights.shape)
    speed = test_curve_speed(npan, aspect)
    #print(speed.shape)
    result = lege_weights * test_curve_speed(npan, aspect)
    #print(result.shape)
    return result.reshape(-1)

def test_curve_weights_teardrop(npan, theta):
    #print(lege_weights.shape)
    speed = test_curve_speed_teardrop(npan, theta)
    #print(speed.shape)
    result = lege_weights * test_curve_speed_teardrop(npan, theta)
    #print(result.shape)
    return result.reshape(-1)

def get_r(t, eps, aspect):
    return get_rp(t+eps, aspect)
def get_rp(t,aspect):
    return np.array([aspect*np.cos(2*np.pi*t),np.sin(2*np.pi*t)])
def get_nup(t,aspect):
    normal = np.array([-np.cos(2*np.pi*t),-aspect*np.sin(2*np.pi*t)])
    return normal / np.linalg.norm(normal, 2)
def K_lim(t, eps, aspect):
    diff = (get_r(t, eps, aspect) - get_rp(t, aspect))
    result = np.dot(get_nup(t, aspect), diff) / (2*np.pi*np.linalg.norm(diff, 2)**2)
    return result
    
def K_eval(nu_p, r, r_p):
    nu_p_v = np.array([nu_p.real, nu_p.imag])
    r_v = np.array([r.real, r.imag])
    r_p_v = np.array([r_p.real, r_p.imag])
    
    result = np.sum(np.dot(nu_p_v, r_v - r_p_v))
    result /= 2*np.pi*np.sum(np.dot(r_v - r_p_v, r_v - r_p_v))
    return result

def compute_double_layer_kernel_test(complex_positions, curve_normal, aspect, npoin, parametrization):
    K = np.empty((npoin,npoin))
    for i in range(npoin):
        for j in range(npoin):
            r = complex_positions[i]
            r_p = complex_positions[j]
            nu_p = curve_normal[j]
            if i == j:
                K[i, j] = K_lim(parametrization[i], 0.001, aspect)
            else:
                K[i, j] = K_eval(nu_p, r, r_p)
    return K

def compute_double_layer_off_boundary(complex_positions, curve_normal, target_complex, npoin):
    OUT = np.empty(npoin)
    for j in range(npoin):
        r = target_complex
        r_p = complex_positions[j]
        nu_p = curve_normal[j]
        
        OUT[j] = K_eval(nu_p, r, r_p)
    return OUT

def get_naive_potential(npan, test_charge, target_complex):
    #Defining Number of Panels
    #npan = int(np.loadtxt('../InitialConditions/npan.np')[1])
    #npan = 40
    #print("Number of Panels: ", npan)
    npoin = npan*16

    #Defining Relevant Parametrized Quantities
    aspect = 3
    panel_boundaries = np.linspace(0, 1, npan+1)
    curve_nodes = ellipse(make_panels(panel_boundaries), stretch=aspect)
    curve_nodes = curve_nodes.reshape(2,-1)
    curve_normal = np.array(ellipse_normal(make_panels(panel_boundaries), aspect, npoin))
    parametrization = make_panels(panel_boundaries).reshape(-1)
    complex_positions = [complex(curve_nodes[0][i],curve_nodes[1][i]) for i in range(npoin)]
    complex_positions = np.array(complex_positions)
    #plt.scatter(complex_positions.real, complex_positions.imag)
    #plt.axis('equal')
    #plt.title('Positions of Nodes')
    #plt.show()

    D_K = compute_double_layer_kernel_test(complex_positions, curve_normal, aspect, npoin, parametrization)
    W_shape = np.diag(test_curve_weights(npan, aspect))
    D_KW = D_K @ W_shape
    LHS = 0.5*np.eye(npoin) + D_KW
    #RHS = np.loadtxt("../InitialConditions/bc_potential.np")
    #test_charge = np.array([-2,2])
    RHS = get_bc_conditions([test_charge], complex_positions)
    #assert(np.max(np.abs(RHS-get_bc_conditions([test_charge], complex_positions)))<=1e-6)

    #density = gmres(LHS, RHS)[0]
    density = np.linalg.solve(LHS, RHS)
    #target_complex =0.5+ complex(0,1)*0
    out = compute_double_layer_off_boundary(complex_positions, curve_normal, target_complex, npoin) @ W_shape @ density   
    #print("OUT:", out)
    true = get_potential(np.array([target_complex.real,target_complex.imag]), [test_charge])

    return out, true


def get_error(npan, test_charge, target_complex):
    #Defining Number of Panels
    #npan = int(np.loadtxt('../InitialConditions/npan.np')[1])
    #npan = 40
    #print("Number of Panels: ", npan)
    npoin = npan*16

    #Defining Relevant Parametrized Quantities
    aspect = 3
    panel_boundaries = np.linspace(0, 1, npan+1)
    curve_nodes = ellipse(make_panels(panel_boundaries), stretch=aspect)
    curve_nodes = curve_nodes.reshape(2,-1)
    curve_normal = np.array(ellipse_normal(make_panels(panel_boundaries), aspect, npoin))
    parametrization = make_panels(panel_boundaries).reshape(-1)
    complex_positions = [complex(curve_nodes[0][i],curve_nodes[1][i]) for i in range(npoin)]
    complex_positions = np.array(complex_positions)
    #plt.scatter(complex_positions.real, complex_positions.imag)
    #plt.axis('equal')
    #plt.title('Positions of Nodes')
    #plt.show()

    D_K = compute_double_layer_kernel_test(complex_positions, curve_normal, aspect, npoin, parametrization)
    W_shape = np.diag(test_curve_weights(npan, aspect))
    D_KW = D_K @ W_shape
    LHS = 0.5*np.eye(npoin) + D_KW
    #RHS = np.loadtxt("../InitialConditions/bc_potential.np")
    #test_charge = np.array([-2,2])
    RHS = get_bc_conditions([test_charge], complex_positions)
    #assert(np.max(np.abs(RHS-get_bc_conditions([test_charge], complex_positions)))<=1e-6)

    #density = gmres(LHS, RHS)[0]
    density = np.linalg.solve(LHS, RHS)
    #target_complex =0.5+ complex(0,1)*0
    out = compute_double_layer_off_boundary(complex_positions, curve_normal, target_complex, npoin) @ W_shape @ density   
    print("OUT:", out)
    true = get_potential(np.array([target_complex.real,target_complex.imag]), [test_charge])

    return np.abs(out-true)



""" def main_teardrop():
    #Defining Number of Panels
    #npan = int(np.loadtxt('../InitialConditions/npan.np')[1])
    theta = np.pi/2

    #Number of panels = 10
    npan = 10
    sinter = np.linspace(0, 1, npan+1)
    sinterdiff = np.ones(npan)/npan
    test_charge = np.array([0,0.4])
    target_complex= 0.5 + complex(0,1)*0.05

    z, zp, zpp, nz, w, wzp, npoin = zinit(theta, sinter, sinterdiff, lege_nodes, lege_weights, npan)
    complex_positions = z
    curve_normal = nz

    plt.scatter(z.real,z.imag)
    plt.scatter(test_charge[0],test_charge[1])
    plt.scatter(target_complex.real, target_complex.imag)
    plt.show()

    W_shape = np.diag(np.abs(wzp))
    D_K = MAinit(z,zp,zpp,nz,w,wzp,npoin)

    D_KW = D_K 
    LHS = 0.5*np.eye(npoin) + D_KW @ W_shape
    RHS = 2*get_bc_conditions([test_charge], z)

    density = gmres(LHS, RHS)[0]

    print(LHS)

    out = compute_double_layer_off_boundary(complex_positions, curve_normal, target_complex, npoin) @ W_shape @ density   

    print("OUT:", out)
    true = get_potential(np.array([target_complex.real,target_complex.imag]), [test_charge])
    print("True:", true) """


def main():
    #Defining Number of Panels
    #npan = int(np.loadtxt('../InitialConditions/npan.np')[1])
    npan = 10
    #print("Number of Panels: ", npan)
    npoin = npan*16

    #Defining Relevant Parametrized Quantities
    aspect = 3
    panel_boundaries = np.linspace(0, 1, npan+1)
    curve_nodes = ellipse(make_panels(panel_boundaries), stretch=aspect)
    curve_nodes = curve_nodes.reshape(2,-1)
    curve_normal = np.array(ellipse_normal(make_panels(panel_boundaries), aspect, npoin))
    parametrization = make_panels(panel_boundaries).reshape(-1)
    complex_positions = [complex(curve_nodes[0][i],curve_nodes[1][i]) for i in range(npoin)]
    complex_positions = np.array(complex_positions)
    #plt.scatter(complex_positions.real, complex_positions.imag)
    #plt.axis('equal')
    #plt.title('Positions of Nodes')
    #plt.show()
    
    #D_K_old = compute_double_layer_kernel_test(complex_positions, curve_normal, aspect, npoin, parametrization)
    sympy_kern = sympy_kernel(3)
    D_K = np.zeros((npoin, npoin))
    for i in range(npoin):
        D_K[i,:] = sympy_kern.kernel_evaluate(parametrization[i],parametrization)
    for i in range(npoin):
        D_K[i,i] = sympy_kern.kernel_evaluate_equal(parametrization[i])


    """ 
    index = np.argmax(D_K - D_K_old)
    print(index // npoin, index % npoin)
    print(np.max(D_K - D_K_old))
    """

    W_shape = np.diag(test_curve_weights(npan, aspect))
    D_KW = D_K @ W_shape
    LHS = 0.5*np.eye(npoin) + D_KW
    #RHS = np.loadtxt("../InitialConditions/bc_potential.np")
    test_charge = np.array([-2,2])
    RHS = get_bc_conditions([test_charge], complex_positions)
    #assert(np.max(np.abs(RHS-get_bc_conditions([test_charge], complex_positions)))<=1e-6)
   
    density = gmres(LHS, RHS)[0]
    #print("Coarse Naive Density:", density)
    #print("Coarse Naive Shape:", density.shape)
    target_complex= 0+ complex(0,1)*0.2
    out = compute_double_layer_off_boundary(complex_positions, curve_normal, target_complex, npoin) @ W_shape @ density   
    print("Result:", out)
    true = get_potential(np.array([0,0.2]), [test_charge])
    print("True:", true)
    print("Error:", np.abs(out-true))


if __name__ == '__main__':
    main()




# %%
