import unittest
from src import grid

#python -m unittest test_module1 test_module2

class DerivativeTests(unittest.TestCase):
    def test_symmetry_second(self):
        dL = [0.01,0.01];
        N = [100,100];
        testgrid = grid.FiniteDifference(dL,N);
        ## check matrix is symmetric
        Dxf = testgrid.Dxf;
        Dyf = testgrid.Dxb;
        Dx2 = Dxf@Dyf;
        self.assertEqual((abs(Dx2-Dx2.T)>1e-10).nnz, 0);

    def testnnz_elems(self):
        return;

if __name__ == '__main__':
    unittest.main()
