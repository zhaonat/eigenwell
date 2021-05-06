from .eigen import *
from .constants import *
from scipy.sparse.linalg import spsolve as bslash

class EigenK2D(Eigen):
    '''
        like eigenguide 2D except it is written as a quadratic eigenvalue problem

        eigenproblem has general form: OA(x) = OB\lambda (x)
    '''
    def __init__(self, structure, grid, omega = 0, polarization = 'TE'):

        super().__init__(structure, grid);
        self.polarization = polarization
        self.make_operator_components(omega);

        return;

    def make_operator_components(self, omega):

        M = self.structure.M;
        N = self.structure.N;
        Dxf = self.grid.Dxf; Dyf = self.grid.Dyf;
        Dxb = self.grid.Dxb; Dyb = self.grid.Dyb;
        I = sp.identity(M);

        Epxx = np.reshape(self.structure.epxx, (M,), order = 'F')
        Epyy = np.reshape(self.structure.epyy, (M,), order = 'F');
        Epzz = np.reshape(self.structure.eps_r, (M,), order = 'F');

        invTepxx = sp.spdiags(1 / Epxx, 0, M,M)
        invTepyy = sp.spdiags(1 / Epyy, 0, M,M)
        invTepzz = sp.spdiags(1 / Epzz, 0, M,M)

        if(self.polarization == 'TM'):
            self.Mop = invTepxx;
            self.Cop = -(-1j * (Dxf@invTepxx + @invTepxx@Dxb));
            self.Kop = (-Dxf @ invTepxx@ Dxb - Dyf @ invTepyy@Dyb) + omega**2*I
            self.Kpart = (-Dxf @ invTepxx@ Dxb - Dyf @ invTepyy@Dyb)
        elif(self.polarization == 'TE'):
            self.Kop = invTepzz@(-Dxf @ Dxb - Dyf @ Dyb) + omega**2*I
            self.Kpart = invTepzz@(-Dxf @ Dxb - Dyf @ Dyb)
            self.Mop = invTepzz;
            self.Cop = -invTepzz@(1j * (Dxf + Dxb)); #% lambda

        OB = sp.bmat([[self.Mop, None],[None, I]]);
        OA = sp.bmat([[self.Cop, K_omega],[-I, None]]);
        self.OA = OA;
        self.OB = OB;

    def update_operator(self,omega):
        '''
            only run after make_operator_components is called
        '''
        self.Kop = self.Kpart+omega**2*sp.identity(self.structure.M)
        OB = sp.bmat([[self.Mop, None],[None, I]]);
        OA = sp.bmat([[self.Cop, Kop],[-I, None]]);
        self.OA = OA;
        self.OB = OB;
