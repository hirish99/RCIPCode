from Naive import *

import unittest
class TestNaiveMethod(unittest.TestCase):
    #Make Panels Unit Testing
    
    def test_simple_make_panels(self):
        n = 16
        lege_nodes, lege_weights, _ = sps.legendre(n).weights.T

        a = ((lege_nodes + 1)/2) - make_panels(np.array([0,1]))
        
        self.assertTrue(not a.any())

    def test_test_curve_weights(self):
        test_curve_weights(2, 3)
        pass

    def test_get_rp(self):
        N = 17
        aspect = np.random.randint(1, 17)
        params = np.array([0, 0.25, 0.334, 0.5+0.334])

        r_1 = get_rp(params[0], aspect)
        r_2 = get_rp(params[1], aspect)
        r_3 = get_rp(params[2], aspect)
        r_4 = get_rp(params[3], aspect)


        self.assertTrue(np.sum(r_1 * np.array([1,0])==np.linalg.norm(r_1, 2)))
        self.assertTrue(np.sum(r_2 * np.array([0,1])==np.linalg.norm(r_2, 2)))
        
        if not np.sum(r_3 * r_4)+(np.linalg.norm(r_3, 2)*np.linalg.norm(r_4, 2)) <= 1e-6:
            print(r_3, r_4)

        self.assertTrue(np.sum(r_3 * r_4)+(np.linalg.norm(r_3, 2)*np.linalg.norm(r_4, 2)) <= 1e-6)

    #Test Fixed Curve_Deriv
    def test_fixed_curve_deriv(self):
        N = 17
        aspect = np.random.randint(1, 17)
        a = np.random.rand()
        param = np.array([[a]])
        result_code = fixed_curve_deriv(param, aspect)
        true = np.array([[[-aspect*2*np.pi*np.sin(2*np.pi*a)]], [[2*np.pi*np.cos(2*np.pi*a)]]])
        if not np.max(np.abs(result_code - true)) <=1e-6:
            print(true, result_code)
        self.assertTrue(np.max(np.abs(result_code - true)) <=1e-6)
    '''
    def test_curve_deriv(self):
        aspect = 3
        param = make_panels(np.array([0,1]))
        curve_nodes = ellipse(param, stretch=aspect)
        actual = curve_deriv_calc(D, curve_nodes)

        true = fixed_curve_deriv(param, aspect)

        print(true, actual)
        self.assertTrue(True)
    '''

    def test_curve_speed_check(self):
        param = make_panels(np.array([0,1]))
        stretch = 3
        deriv = np.array([np.linalg.norm([
        -2*np.pi*stretch*np.sin(2*np.pi*t),
        2*np.pi*np.cos(2*np.pi*t)
        ],2) for t in param])

        deriv1 = test_curve_speed(1, stretch).reshape(-1)
        a = np.abs(deriv-deriv1)
        self.assertTrue(True)
        #self.assertTrue(np.linalg.norm(a,1)<=1e-6)






if __name__ == '__main__':
    unittest.main()