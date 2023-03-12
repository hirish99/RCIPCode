from RCIP import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import unittest

class TestNaiveMethod(unittest.TestCase):

    #function on the parametrization as well as the 
    def f_param(self, param, z_in):
        result = z_in.real

        return result

    def test_prolongation_error_ellipse(self):
        npan = 10
        nsub = 30
        aspect = 3

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

        print("Difference of Prolongation:",np.linalg.norm(f_fine - P @ f_coarse, 2))


if __name__ == '__main__':
    unittest.main()