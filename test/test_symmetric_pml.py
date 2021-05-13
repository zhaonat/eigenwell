import unittest
import numpy as np
from eigenwell.src import grid
from eigenwell.src.pml import *
from eigenwell.src.constants import *
#python -m unittest test_module1 test_module2

'''
    make sure the derivative test ALSO PASSES
'''

class SymmetricPMLTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        '''
            units are in microns
        '''
        super(SymmetricPMLTests, self).__init__(*args, **kwargs)
        self.N = [100,100];
        wvlen = 0.2e-6;
        self.Npml = [15,15]
        self.omega = 2*np.pi*C0/wvlen;
        L = np.array([1e-6, 1e-6]);
        self.dL= np.array(L)/np.array(self.N)
        self.xrange = [-L[0]/2, L[0]/2]
        self.yrange = [-L[1]/2, L[1]/2]
        self.testgrid = grid.FiniteDifferenceGrid(self.dL,self.N);

    def test_symmetry_TE(self):
        ## check matrix is symmetric
        symmetric_pml = SymmetrizePML(self.N, self.Npml, self.omega,polarization = 'TE');
        symmetric_pml.Soperators(self.xrange,self.yrange)

        Sxf, Syf = symmetric_pml.Sxf, symmetric_pml.Syf;
        Sxb, Syb = symmetric_pml.Sxb, symmetric_pml.Syb;

        Dxf = self.testgrid.Dxf; Dxb = self.testgrid.Dxb;
        Dyf = self.testgrid.Dyf; Dyb = self.testgrid.Dyb;

        A = symmetric_pml.Pl@((Sxb@Dxb)@(Sxf@Dxf) + (Syb@Dyb)@(Syf@Dyf))@symmetric_pml.Pr
        self.assertEqual(((np.abs(A-A.T)/np.max(np.abs(A)))>1e-10).nnz, 0);


    def test_symmetry_TM(self):
        M = np.prod(self.N)
        eps_r = np.ones(self.N)
        eps_r[40:60,40:60] = 12;
        invTepxx = sp.spdiags(1/(EPSILON0*eps_r.flatten(order = 'F')), 0, M,M)
        invTepyy = sp.spdiags(1/(EPSILON0*eps_r.flatten(order = 'F')), 0, M,M)
        symmetric_pml = SymmetrizePML(self.N, self.Npml, self.omega,polarization = 'TM');
        symmetric_pml.Soperators(self.xrange, self.yrange)

        Dxf = self.testgrid.Dxf; Dxb = self.testgrid.Dxb;
        Dyf = self.testgrid.Dyf; Dyb = self.testgrid.Dyb;
        Sxf, Syf = symmetric_pml.Sxf, symmetric_pml.Syf;
        Sxb, Syb = symmetric_pml.Sxb, symmetric_pml.Syb;

        A = -(1/MU0)* ( Sxf@Dxf@(invTepxx)@Sxb@Dxb + Syf@Dyf@(invTepyy)@Syb@Dyb )
        A = symmetric_pml.Pl@A@symmetric_pml.Pr
        self.assertEqual((np.abs(A-A.T)/np.max(np.abs(A))>1e-10).nnz, 0);



if __name__ == '__main__':
    unittest.main()
