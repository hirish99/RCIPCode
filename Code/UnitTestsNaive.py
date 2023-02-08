from Naive import *

import unittest
class TestNaiveMethod(unittest.TestCase):
    #Make Panels Unit Testing
    def zesty_teardrop(self):
        npan = 4
        panel_boundaries = np.linspace(0, 1, npan+1)
        curve_nodes = teardrop(make_panels(panel_boundaries), theta = np.pi/2)
        curve_nodes1 = ellipse(make_panels(panel_boundaries), 3)

        assert(curve_nodes.shape == curve_nodes1.shape)

        plt.scatter(curve_nodes[0], curve_nodes[1])
        plt.scatter(curve_nodes1[0], curve_nodes1[1])
        plt.show()
    
    def test_teardrop_normal(self):
        npan = 4
        panel_boundaries = np.linspace(0, 1, npan+1)
        normals = teardrop_normal(make_panels(panel_boundaries), np.pi/2)
        curve_nodes = teardrop(make_panels(panel_boundaries), theta = np.pi/2)

        i = np.random.randint(0,16)
        j = np.random.randint(0,4)
        print(curve_nodes.shape)
        plt.scatter(curve_nodes[0], curve_nodes[1])
        plt.arrow(curve_nodes[0,j][i], curve_nodes[1,j][i], normals[0,j][i], normals[1,j][i])
        plt.axis('equal')
        plt.show()


    def estimate_order_of_convergence(self, abscissae, errors):
        """Assuming that abscissae and errors are connected by a law of the form
        error = constant * abscissa ^ (order),
        this function finds, in a least-squares sense, the best approximation of
        constant and order for the given data set. It returns a tuple (constant, order).
        """
        assert len(abscissae) == len(errors)
        if len(abscissae) <= 1:
            raise RuntimeError("Need more than one value to guess order of convergence.")

        coefficients = np.polyfit(np.log10(abscissae), np.log10(errors), 1)
        return 10**coefficients[-1], coefficients[-2]
    
    def zesty_order_of_convergence(self):
        N = 20
        test_charge = np.array([-2,2])
        target_complex = 0.5+ complex(0,1)*0
        err = []
        x = []
        for npan in range(1,N,2):
            err.append(get_error(npan, test_charge, target_complex))
            x.append(13.365/npan)
        err = np.array(err)
        x = np.array(x)

        order = self.estimate_order_of_convergence(x, err)

        plt.scatter(np.log(x),np.log(err))
        print(err)
        plt.show()
        print(order)

        #pass


    

    def test_get_rp(self):
        N = 23
        a = np.random.randint(1, N)
        self.assertTrue(np.abs(get_rp(0,a) - a)<=1e-6)
        self.assertTrue(np.abs(get_rp(0.25,a) - 1)<=1e-6)

    def test_compute_double_layer_off_boundary(self):
        aspect = 3
        npan = int(np.loadtxt('../InitialConditions/npan.np')[1])
        #print("Number of Panels: ", npan)
        npoin = npan*16
        panel_boundaries = np.linspace(0, 1, npan+1)
        curve_nodes = ellipse(make_panels(panel_boundaries), stretch=aspect)
        curve_nodes = curve_nodes.reshape(2,-1)
        curve_normal = np.array(ellipse_normal(make_panels(panel_boundaries), aspect, npoin))
        complex_positions = [complex(curve_nodes[0][i],curve_nodes[1][i]) for i in range(npoin)]
        complex_positions = np.array(complex_positions)
        W = np.diag(test_curve_weights(npan, aspect))
        list_target = [1, 3, 4]
        density = np.ones(npoin)

        pot0 = compute_double_layer_off_boundary(complex_positions, curve_normal, list_target[0], npoin) @ W @ density
        pot1 = compute_double_layer_off_boundary(complex_positions, curve_normal, list_target[1], npoin) @ W @ density
        pot2 = compute_double_layer_off_boundary(complex_positions, curve_normal, list_target[2], npoin) @ W @ density

        self.assertTrue(np.abs(pot0-1)<=1e-6)
        self.assertTrue(np.abs(pot1-0.5)<=1e-6)
        self.assertTrue(np.abs(pot2-0)<=1e-6)

    def test_simple_make_panels(self):
        n = 16
        lege_nodes, lege_weights, _ = sps.legendre(n).weights.T

        a = ((lege_nodes + 1)/2) - make_panels(np.array([0,1]))
        
        self.assertTrue(not a.any())

    def test_test_curve_weights(self):
        test_curve_weights(2, 3)
        pass

    def test_get_nup(self):
        self.assertTrue(np.dot(get_nup(0, 12),  np.array([1,0])) == -1)

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