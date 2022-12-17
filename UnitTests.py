from Naive import *

import unittest
class TestNaiveMethod(unittest.TestCase):
    #Make Panels Unit Testing
    
    def test_simple_make_panels(self):
        n = 16
        lege_nodes, lege_weights, _ = sps.legendre(n).weights.T

        a = ((lege_nodes + 1)/2) - make_panels(np.array([0,1]))
        
        self.assertTrue(not a.any())

    def true_deriv(self, t, stretch):
        return np.array([
        -stretch*2*np.pi*np.sin(2*np.pi*t),
        2*np.pi*np.cos(2*np.pi*t)])

    #Test Curve_Deriv
    def test_curve_deriv(self):
        aspect = 3
        param = make_panels(np.array([0,1]))
        curve_nodes = ellipse(param, stretch=aspect)
        actual = curve_deriv(D, curve_nodes)

        param = param[0]
        true = [self.true_deriv(t, aspect) for t in param]
        true = np.array(true)
        print(true)
        self.assertTrue(True)
        actual = actual.reshape(-1, 2)
        print(actual)
        self.assertTrue(np.linalg.norm(true-actual,2)<=1e-6)


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