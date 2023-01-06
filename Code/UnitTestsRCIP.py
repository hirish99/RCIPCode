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

    def test_zinit_ellipse(self):
        s_2, w_2 = zloc_init_ellipse(T, W, 4, 2, 3)
        s_1, w_1 = zloc_init_ellipse(T, W, 4, 1, 3)

        print(s_2.min(), s_2.max())
        print(s_1.min(), s_1.max())
        print(s_1.shape)
        print(s_2.shape)
        plt.scatter(np.arange(0,96), s_2, label="s_2")
        plt.scatter(np.arange(0,96), s_1, label="s_1")
        plt.legend()
        plt.show()



   
    def test_zloc_init_ellipse(self):
        pass





        
        





if __name__ == '__main__':
    unittest.main()