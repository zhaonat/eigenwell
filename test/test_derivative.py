import unittest
import numpy as np

from eigenwell.src import grid
#python -m unittest test_module1 test_module2

class DerivativeTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DerivativeTests, self).__init__(*args, **kwargs)
        self.dL = [0.01,0.01];
        self.N = [100,100];
        self.testgrid = grid.FiniteDifferenceGrid(self.dL,self.N);

    def test_symmetry_second(self):
        ## check matrix is symmetric
        Dxf = self.testgrid.Dxf;
        Dxb = self.testgrid.Dxb;
        Dx2 = Dxf@Dxb;
        Dy2 = self.testgrid.Dyf@self.testgrid.Dyb;

        self.assertEqual((abs(Dx2-Dx2.T)>1e-10).nnz, 0);
        self.assertEqual((abs(Dy2-Dy2.T)>1e-10).nnz, 0);

    def testnnz_elems(self):
        Dxf = self.testgrid.Dxf;
        Dxb = self.testgrid.Dxb;
        Dx2 = Dxf@Dxb;
        Dyf = self.testgrid.Dyf;
        Dyb = self.testgrid.Dyb;
        Dy2 = Dyf@Dyb;
        M = Dx2.shape[0]
        self.assertEqual(Dx2.count_nonzero(),3*M);
        self.assertEqual(Dy2.count_nonzero(),3*M);

    def dxf_to_dxb(self):
        '''
            check dxf = -dxb.T;
            check dyf = -dyb.T;
        '''
        Dxf = self.testgrid.Dxf;
        Dyf = self.testgrid.Dxb;
        Dyf = self.testgrid.Dyf;
        Dyb = self.testgrid.Dyb;
        self.assertEqual((abs(Dxf+Dxb.T)>1e-10).nnz, 0);
        self.assertEqual((abs(Dyf+Dyb.T)>1e-10).nnz, 0);

        return;

if __name__ == '__main__':
    unittest.main()
