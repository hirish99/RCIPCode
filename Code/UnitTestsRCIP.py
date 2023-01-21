from RCIP import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import unittest

class TestNaiveMethod(unittest.TestCase):





    def test_check_give_fine_mesh(self):
        nsub = 2
        npan = 10
        aspect = 1
        s, w_fine, kcirc = give_fine_mesh_parametrization_ellipse(nsub, npan)

        #print("Sum of Weights:", np.sum(w_fine))

        awzp = np.abs(zpfunc_ellipse(s, aspect))

        self.assertTrue(np.sum(awzp * w_fine) == 2*np.pi)




    def test_verify_rho_fine(self):
        #FINE DENSITY COMPUTATION 
        nsub = 2
        npan = 10
        aspect = 3

        K_fine = get_K_fine(nsub, npan, aspect)

        s, w_fine = give_fine_mesh_parametrization_ellipse(nsub, npan)[0:2]
        z = ellipse(s, aspect)
        z = z[0,:]+complex(0,1)*z[1,:]

        test_charge = np.array([-2,2])
        RHS_fine = 2*get_bc_conditions([test_charge], z) 

        LHS_fine = np.eye(K_fine.shape[0]) + K_fine
        density_fine = gmres(LHS_fine, RHS_fine)[0]
        
        param, weights, kcirc = give_fine_mesh_parametrization_ellipse(nsub, npan)
        true_density = get_density_true(param, weights, aspect)
        print(true_density-density_fine)

    def test_eq_7_RCIP_LONG(self):
        #FINE DENSITY COMPUTATION 
        nsub = 2
        npan = 10
        aspect = 3
        K_fine = get_K_fine(nsub, npan, aspect)

        s, w_fine = give_fine_mesh_parametrization_ellipse(nsub, npan)[0:2]
        z = ellipse(s, aspect)
        z = z[0,:]+complex(0,1)*z[1,:]

        test_charge = np.array([-2,2])
        RHS_fine = 2*get_bc_conditions([test_charge], z) 

        LHS_fine = np.eye(K_fine.shape[0]) + K_fine
        density_fine = gmres(LHS_fine, RHS_fine)[0]

        target = np.array([0,0.2])

        z_list = np.empty((K_fine.shape[0],2))
        z_list[:,0] = z.real
        z_list[:,1] = z.imag
        f_fine = f(z_list, target)

        #print(f_fine)

        # YOU REALLY NEED TO WRITE A UNIT TEST VERIFYING DENSITY_FINE!!!
        #print(np.sum(density_fine * w_fine))

    def test_eq_15(self):

        #FINE DENSITY COMPUTATION 
        nsub = 2
        npan = 10
        aspect = 3
        K_fine = get_K_fine(nsub, npan, aspect)

        s = give_fine_mesh_parametrization_ellipse(nsub, npan)[0]
        z = ellipse(s, aspect)
        z = z[0,:]+complex(0,1)*z[1,:]

        test_charge = np.array([-2,2])
        RHS_fine = 2*get_bc_conditions([test_charge], z) 

        LHS_fine = np.eye(K_fine.shape[0]) + K_fine
        density_fine = gmres(LHS_fine, RHS_fine)[0]
        #FINE DENSITY COMPUTATION 

        #Transformed Density Computation
        IP, IPW = IPinit(T,  W)

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
        #R = get_R_true(npan, nsub, aspect)


        I_coa = np.eye(npoin)

        LHS = I_coa + (Kcirc@R)
        #pot_boundary = np.loadtxt('bc_potential.np')
        
        test_charge = np.array([-2,2])
        RHS = 2*get_bc_conditions([test_charge], z)

        density_tilde = gmres(LHS, RHS)[0]

        ## CHECK

        P_true = get_P(npan, nsub)

        K_star_fine = get_K_star_circ_fine(nsub, npan, aspect)[0]

        #print(density_fine.shape)
        #print(K_star_fine.shape)

        #print((np.eye(K_star_fine.shape[0])+K_star_fine)@ density_fine - P_true @ density_tilde)
        #print(np.mean(P_true @ density_tilde))






    def test_LHS(self):
        pass

    def test_R_comp(self):
        npan = 10
        nsub = 2
        aspect = 3
        IP, IPW = IPinit(T,  W)
        Pbc = block_diag(np.eye(16),IP,IP,np.eye(16))
        PWbc = block_diag(np.eye(16),IPW,IPW,np.eye(16))
        R_sp = Rcomp_ellipse(aspect,T,W,Pbc,PWbc,nsub,npan)


        s, w = zinit_ellipse(T,  W, npan)
        z = zfunc_ellipse(s, aspect)
        z = z[0]
        npoin = s.shape[0]

        #In the paper K absorbs a factor of 2, my MAinit_ellipse doesn't have that factor of 
        starind = [i for i in range(npoin-32,npoin)]
        starind += [i for i in range(32)]

        R = np.eye(npoin)
        #Not the most efficient but quadratic in the order of quadrature
        l=0
        for i in starind:
            m=0
            for j in starind:
                R[i,j] = R_sp[l,m]
                m+=1
            l+=1
        
        R_true = get_R_true(npan, nsub, aspect)

        #plt.title("Difference Between R/R_true")
        #plt.imshow(np.log(np.abs(R-R_true)+1e-15))
        #plt.colorbar()
        #plt.show()


    def test_get_R_true(self):
        npan = 10
        nsub = 1
        aspect = 3
        R = get_R_true(npan, nsub, aspect)

        """ plt.title("Log Plot of R")
        plt.imshow(np.log(1e-16+np.abs(R)))
        plt.colorbar()
        plt.show() """

        """ kcirc = give_fine_mesh_parametrization_ellipse(nsub, npan)[2]


        #I'm just verifying that R only has a single full block 64x64
        max_difference = 0
        i_max = 0
        j_max = 0
        max_differences = []
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if (i in kcirc) and (j in kcirc):
                    if i != j and np.abs(R[i,j]) > max_difference:
                        max_difference = np.abs(R[i,j])
                        i_max = i
                        j_max = j
                        max_differences.append(R[i,j])

        print(i_max, j_max) """
        #TO DO: FIGURE OUT WHY I_MAX AND J_MAX ARE VERY NON-ZERO

    def test_andreas_jan_17_log_plot_inv(self):
        npan = 10
        nsub = 2
        aspect = 3
        Kstar_fine = get_K_star_circ_fine(nsub, npan, aspect)[0]
        P = get_P(npan, nsub)
        PW_T = get_PW(npan, nsub).T
        npoin = Kstar_fine.shape[0]

        """ plt.title("Log Plot of inv(I+K*)")
        plt.imshow(np.log(1e-16+np.abs(np.linalg.inv(np.eye(npoin) +  Kstar_fine))))
        plt.colorbar()
        plt.show() """

        """ plt.title("PW.T")
        plt.imshow(PW_T)
        plt.colorbar()
        plt.show() """

        """ plt.title("P")
        plt.imshow(P)
        plt.colorbar()
        plt.show() """

    def test_andreas_jan_17_log_plot(self):
        npan = 10
        nsub = 2
        aspect = 3

        Kstar_fine = get_K_star_circ_fine(nsub, npan, aspect)[0]
        #plt.title("Log Plot of Kstar_fine")
        #plt.imshow(np.log(1e-16+np.abs(Kstar_fine)))
        #plt.colorbar()
        #plt.show()
    """ 
    def test_fine_again(self):
        nsub = 2
        npan = 10
        print("START")
        param,weights,kcirc = give_fine_mesh_parametrization_ellipse(nsub, npan)
        print(kcirc)
        print("END")


    def test_get_R_true(self):

        npan = 10
        nsub = 1
        aspect = 3
        R = get_R_true(npan, nsub, aspect)

        kcirc = give_fine_mesh_parametrization_ellipse(nsub, npan)[2]


        #I'm just verifying that R only has a single full block 64x64
        max_difference = 0
        i_max = 0
        j_max = 0
        max_differences = []
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if (i in kcirc) and (j in kcirc):
                    if i != j and np.abs(R[i,j]) > max_difference:
                        max_difference = np.abs(R[i,j])
                        i_max = i
                        j_max = j
                        max_differences.append(R[i,j])

        print(max_difference, i_max, j_max, max_differences)
                    

    """


    def test_eq_16(self):
        nsub = 2
        npan = 10
        aspect = 3
        K_circ_fine = get_K_star_circ_fine(nsub, npan, aspect)[1]
        K_circ_coarse = get_K_circ_coarse(npan, aspect)
        P = get_P(npan, nsub)
        PW = get_PW(npan, nsub)
        """ print(np.max(K_circ_fine - P@K_circ_coarse@PW.T)) """

    def test_PW_P(self):
        npan = 10
        PW = get_PW(npan,2)
        P = get_P(npan,2)
        """ print(np.max(PW.T @ P - np.eye(16*npan))) """

    def test_get_PW(self):
        PW = get_PW(10,2)
        """ plt.imshow(PW)
        plt.show() """

    def test_get_P(self):
        """         plt.figure(3)
        plt.title("True 64->128")
        P = get_P_helper(6) @ get_P_helper(4)
        plt.imshow(P)

        plt.figure(4)
        plt.title("10 Panels")
        plt.imshow(get_P(10,2))
        print("P SHAPE:", get_P(10,2).shape) """


    def test_get_P_helper(self):
        """         plt.figure(1)
        plt.title("Prolongation 64->96")
        plt.imshow(get_P_helper(4))
        plt.figure(2)
        plt.title("Prolongation 96->128")
        plt.imshow(get_P_helper(6)) """



    def test_plot(self):
        IP, IPW = IPinit(T,  W)

        theta = np.pi/2
        lamda = 1

        #Number of panels = 10
        npan = 10
        sinter = np.linspace(0, 1, npan+1)
        sinterdiff = np.ones(npan)/npan
        nsub = 6

        #z, zp, zpp, nz, w, wzp = zloc_init(theta, T, W, nsub, 2, npan)
        z, zp, zpp, nz, w, wzp, npoin = zinit(theta, sinter, sinterdiff, T, W, npan)


        """ plt.scatter(z.real, z.imag, c='blue')
        plt.scatter(z.real[:20], z.imag[:20],c='orange')
        plt.scatter(z.real[-20:], z.imag[-20:],c='green')
        plt.show() """



    def initialize_s(self, npan):
        npoin = 16*npan
        s = np.zeros(npoin)

        sinter = np.linspace(0, 1, npan+1)
        sinterdiff = np.ones(npan)/npan

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
        
        return s

    def test_zfunc_ellipse(self):
        s = self.initialize_s(10)
        aspect = np.random.randint(1, 4)
        out = (zfunc_ellipse(s, aspect))

        self.assertTrue(out.real[0,0]-aspect <=1e-3)
        """ print(out.real)
        plt.scatter(out.real[0,:10], out.imag[0,:10])
        plt.title("zfunc")
        plt.show() """

    def test_give_fine_mesh(self):
        npan = 10
        nsub=2
        param3, weights3, kcirc= give_fine_mesh_parametrization_ellipse(nsub, npan)
        ellipse3 = zfunc_ellipse(param3, 3)
        #plt.scatter(np.arange(0,len(weights3)), param3)
        #plt.scatter(np.arange(0,len(weights2)),100*weights2)
        #plt.scatter(np.arange(0,len(weights3)),100*weights3)


        #print(len(weights3),len(param3))
        #plt.scatter((ellipse3.real), ellipse3.imag,c='blue')
        #plt.scatter((ellipse3.real)[0,kcirc], ellipse3.imag[0,kcirc],c='orange')
       
        #plt.show()

    def test_K_star_fine(self):
        nsub=2
        npan=8
        aspect=3

        Kstar,Kcirc = get_K_star_circ_fine(nsub, npan, aspect)

        """         plt.figure(1)
        plt.title("Kstar")
        plt.imshow(Kstar)


        plt.figure(2)
        plt.title("Kcirc")
        plt.imshow(Kcirc) """



    def test_zpfunc_ellipse(self):
        s = self.initialize_s(10)
        aspect = np.random.randint(1, 4)
        out = (zpfunc_ellipse(s, aspect))
        """ plt.scatter(out.real[0,:10], out.imag[0,:10])
        plt.title("zpfunc")
        plt.show() """
        self.assertTrue(out.imag[0,0]-2*np.pi<=1e-8)
        self.assertTrue(out.real[0,0]-0<=1e-8)
    
    def test_zppfunc_ellipse(self):
        s = self.initialize_s(10)
        aspect = np.random.randint(1, 4)
        out = (zppfunc_ellipse(s, aspect))
        self.assertTrue(out.imag[0,0]-0<=1e-8)
        self.assertTrue(out.real[0,0]-aspect*(2*np.pi)**2<=1e-8)

    def test_zloc_init_ellipse(self):
        self.assertTrue(True)

           
        nsub = 2
        npan = 10
        s_2, w_2 = zloc_init_ellipse(T, W, nsub, 2, npan)
        s_1, w_1= zloc_init_ellipse(T, W, nsub, 0, npan)
        """ 
        print(s_2.min(), s_2.max())
        print(s_1.min(), s_1.max())
        print(s_1.shape)
        print(s_2.shape)
        plt.scatter(np.arange(0,96), s_2, label="s_2")
        plt.scatter(np.arange(0,96), s_1, label="s_1")
        plt.legend()
        plt.show()  """
        

    def test_zinit_ellipse(self):
        """         npan = 10
        s, w= zinit_ellipse(T, W, npan)
        plt.scatter(np.arange(0,len(s)),s, label="W")
        plt.scatter(np.arange(0,len(w)),w, label="W")
        plt.show """

    def test_MAinit_ellipse(self):

        npan = 5
        aspect = 3
        s, w = zinit_ellipse(T, W, npan)
        M, M_K, W_new = MAinit_ellipse_check(s, w, aspect)

        #DEPENDENCIES FOR CODE:
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

        sympy_kern = sympy_kernel(3)
        D_K = np.zeros((npoin, npoin))
        for i in range(npoin):
            D_K[i,:] = sympy_kern.kernel_evaluate(parametrization[i],parametrization)
        for i in range(npoin):
            D_K[i,i] = sympy_kern.kernel_evaluate_equal(parametrization[i])
        W_shape = np.diag(test_curve_weights(npan, aspect))
        D_KW = D_K @ W_shape

        #print("M_K", M_K[:2, :2])
        #print("D_K", D_K[:2, :2])
        i = np.random.randint(1,npoin-5)
        #print(np.max(np.abs(M_K[i:i+2, i:i+2] - D_K[i:i+2, i:i+2])))
        self.assertTrue(np.max(np.abs(M_K[i:i+2, i:i+2] - D_K[i:i+2, i:i+2])) <= 1e-8)

        self.assertTrue(np.max(np.abs(np.diag(W_new)-np.diag(W_shape)))<=1e-8)




        
        





if __name__ == '__main__':
    unittest.main()