from RCIP import *

import unittest

class TestNaiveMethod(unittest.TestCase):

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

        """         
        nsub = 5
        npan = 10
        s_2, w_2, denom = zloc_init_ellipse(T, W, nsub, 4, npan)
        s_1, w_1, denom1= zloc_init_ellipse(T, W, nsub, 1, npan)
        print("DENOM:", denom, denom1)

        print(s_2.min(), s_2.max())
        print(s_1.min(), s_1.max())
        print(s_1.shape)
        print(s_2.shape)
        plt.scatter(np.arange(0,96), s_2, label="s_2")
        plt.scatter(np.arange(0,96), s_1, label="s_1")
        plt.legend()
        plt.show() 
        """

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


    def test_Rcomp_ellipse(self):
        IP, IPW = IPinit(T,  W)
        Pbc = block_diag(np.eye(16),IP,IP,np.eye(16))
        PWbc = block_diag(np.eye(16),IPW,IPW,np.eye(16))
        aspect = 3
        nsub = 0
        npan = 10
        R = Rcomp_ellipse(aspect, T, W, Pbc, PWbc, nsub, npan)
        print(R[:2,:2])
        #I cant think of a good test here??


        
        





if __name__ == '__main__':
    unittest.main()