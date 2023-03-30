from RCIP import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import unittest

class TestNaiveMethod(unittest.TestCase):

    def test_convergence_of_KcircR_old_new(self):
        npan = 10

        old_arr = []
        new_arr = []
        x = []

        for i in range(10, 30, 1):
            old_arr.append(np.linalg.norm(old_rcip_problem_KcircR(npan, i)-old_rcip_problem_KcircR(npan, i-1))/np.linalg.norm(old_rcip_problem_KcircR(npan, i)))
            new_arr.append(np.linalg.norm(teardrop_rcip_improved_KcircR(npan, i)-teardrop_rcip_improved_KcircR(npan, i-1))/np.linalg.norm(teardrop_rcip_improved_KcircR(npan, i)))
            x.append(i)

        plt.loglog(x, old_arr, 'o',label='old rcip code')
        plt.loglog(x, new_arr, 'o',label='new rcip code')
        plt.title("KcircR old kernel vs new kernel")
        plt.legend()
        plt.show()

    def test_convergence_of_LHS(self):
        npan = 10

        old_arr = []
        new_arr = []
        x = []

        for i in range(10, 30, 1):
            old_arr.append(np.linalg.norm(old_rcip_problem_LHS_RHS(npan, i)[0]-old_rcip_problem_LHS_RHS(npan, i-1)[0])/np.linalg.norm(old_rcip_problem_LHS_RHS(npan, i)[0]))
            new_arr.append(np.linalg.norm(teardrop_rcip_improved_LHS_RHS(npan, i)[0]-teardrop_rcip_improved_LHS_RHS(npan, i-1)[0])/np.linalg.norm(teardrop_rcip_improved_LHS_RHS(npan, i)[0]))
            x.append(i)

        plt.loglog(x, old_arr, 'o',label='old rcip code')
        plt.loglog(x, new_arr, 'o',label='new rcip code')
        plt.title("LHS old kernel vs new kernel")
        plt.legend()
        plt.show()

    def test_convergence_of_density(self):
        npan = 10

        old_arr = []
        new_arr = []
        x = []

        for i in range(10, 30, 1):
            old_arr.append(np.linalg.norm(old_rcip_problem_density(npan, i)[0]-old_rcip_problem_density(npan, i-1)[0])/np.linalg.norm(old_rcip_problem_density(npan, i)[0]))
            new_arr.append(np.linalg.norm(teardrop_rcip_improved_density(npan, i)[0]-teardrop_rcip_improved_density(npan, i-1)[0])/np.linalg.norm(teardrop_rcip_improved_density(npan, i)[0]))
            x.append(i)

        plt.loglog(x, old_arr, 'o',label='old rcip code')
        plt.loglog(x, new_arr, 'o',label='new rcip code')
        plt.title("rho tilde old kernel vs knew kernel")
        plt.legend()
        plt.show()

    
    def test_convergence_of_R(self):

        theta = np.pi/2
        T, W, _ = sps.legendre(n).weights.T
        IP, IPW = IPinit(T,  W)
        Pbc = block_diag(np.eye(16),IP,IP,np.eye(16))
        PWbc = block_diag(np.eye(16),IPW,IPW,np.eye(16))

        npan = 10

        Rcomp_old_arr = []
        Rcomp_new_arr = []
        x = []

        for i in range(10, 30, 1):
            Rcomp_old_arr.append(np.linalg.norm(Rcomp_teardrop_improved(theta, T, W, Pbc, PWbc, i, npan)-Rcomp_teardrop_improved(theta, T, W, Pbc, PWbc, i-1, npan)))
            Rcomp_new_arr.append(np.linalg.norm(Rcomp(theta, T, W, Pbc, PWbc, i, npan)-Rcomp(theta, T, W, Pbc, PWbc, i-1, npan)))
            x.append(i)

        plt.loglog(x, Rcomp_old_arr, 'o',label='old rcip code')
        plt.loglog(x, Rcomp_new_arr, 'o',label='new rcip code')
        plt.title("R convergence old kernel vs knew kernel")
        plt.legend()
        plt.show()


    #function on the parametrization as well as the 
    def f_param(self, param, z_in):
        #result = z_in.real #original paper expression
        result = param*0 + 1 #constant value? shouldn't there be no error?
        #result = compute_f_true(param, 0.1 + 0*complex(0,1),3)
        return result
    
    def test_plot_convergence(self):
        x = []
        array = []
        array_new = []

        npan = 10

        for i in range(5, 30, 1):
            #array_new.append(self.get_error_teardrop_rcip_improved_test(npan, i)-self.get_error_teardrop_rcip_improved_test(npan, i-1))
            #array.append(np.abs(old_rcip_problem(npan, i)))
            #array_new.append((get_error_teardrop_rcip_improved(npan, i)))
            #array_new.append((get_error_teardrop_rcip(npan, i)))
            x.append((i))
        
        #plt.loglog(x, array, 'o',label='old rcip code')
        #plt.loglog(x, array_new,'o',label='new teardrop code')
        #plt.legend()
        #plt.title("Self-Convergence of Problem w/ Singularity")
        #plt.xlabel("nsub (npan=10)")
        #plt.ylabel("rel. error")
        #plt.show()  


    def get_error_teardrop_rcip_improved_test(self, npan, nsub):
        T, W, _ = sps.legendre(n).weights.T
        IP, IPW = IPinit(T,  W)

        theta = np.pi/2

        #Number of panels = 10
        s, _ = zinit_ellipse(T,  W, npan)
        sinter = np.linspace(0, 1, npan+1)
        sinterdiff = np.ones(npan)/npan

        z, zp, zpp, nz, w, wzp, npoin = zinit(theta, sinter, sinterdiff, T, W, npan)

        #In the paper K absorbs a factor of 2, my MAinit_ellipse doesn't have that factor of 2
        Kcirc = MAinitDL(z,zp,zpp,nz,w,wzp,npoin)

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
        R_sp = Rcomp_teardrop_improved(theta,T,W,Pbc,PWbc,nsub,npan)

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
        #f_list = compute_f_true_teardrop(z, complex(0,1)*zp/np.abs(zp), target_complex)
        f_list = self.f_param(s, z_list)

        awzp = w * np.abs(zpfunc(s, theta))

        pot_at_target = np.sum(f_list*density_hat*awzp)

        

        #print(pot_at_target-true)
        return np.abs(pot_at_target)
    

    def test_prolongation_error_ellipse(self):
        npan = 10
        aspect = 3


        nsub_list = []
        error_list = []

        for nsub in range(2, 30):
            #get interpolation operator here
            P = get_P(npan, nsub)

            param_fine, w_fine, kcirc_indices = give_fine_mesh_parametrization_ellipse(nsub, npan)
            z_fine = zfunc_ellipse(param_fine, aspect)
            z_fine = z_fine[0]
            f_fine = self.f_param(param_fine, z_fine)

            s_coarse, w_coarse = zinit_ellipse(T,  W, npan)
            z_coarse = zfunc_ellipse(s_coarse, aspect)
            z_coarse = z_coarse[0]
            f_coarse = self.f_param(s_coarse, z_coarse)

            #print("Difference of Prolongation:",np.linalg.norm(f_fine - P @ f_coarse, 2))
            nsub_list.append(nsub)
            error_list.append(np.linalg.norm(f_fine - P @ f_coarse))

        
        
        plt.title("Log Scaled Plot of Error vs NSUB")
        plt.xlabel("Number of Subdivisions")
        plt.ylabel("Error")
        plt.loglog(nsub_list, error_list,'o' )
        plt.show()
        

if __name__ == '__main__':
    unittest.main()