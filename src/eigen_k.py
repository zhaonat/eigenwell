from .eigen import *
from .constants import *
from scipy.sparse.linalg import spsolve as bslash

class EigenK1D(Eigen):
    '''
        eps_r: numpy array of the structure
        grid: grid object storing the derivative operators
        axis is in x direction
    '''

    def __init__(self, structure, grid, polarization = 'TE'):

        super().__init__(structure, grid);
        self.polarization = polarization
        self.make_operator_components();

        return;

    def make_operator_components(self, omega):
        '''
            solves for Kx eigenvalue, Ky can be added in using bloch
        '''
        M = self.structure.M;
        N = self.structure.N;
        Dxf = self.grid.Dxf; Dyf = self.grid.Dyf;
        Dxb = self.grid.Dxb; Dyb = self.grid.Dyb;

        invTepzz = sp.spdiags(1 / self.eps_r.flatten(), 0, self.M,self.M)
        if(self.polarization == 'TM'):
            self.Mop = invTepzz;
            self.Cop = -invTepzz@(-1j * (Dxf + Dxb));
            self.Kop = -invTepzz@(Dxf @ Dxb) #- omega**2*mu0*eps0*I;

        elif(self.polarization == 'TE'):
            self.Mop = invTepzz;
            self.Cop = -invTepzz@(-1j * (self.grid.Dxf + self.grid.Dxb));
            self.Kop = -invTepzz@(self.grid.Dxf @ self.grid.Dxb) #- omega**2*mu0*eps0*I;

        return;


class EigenK2D(Eigen):
    '''
        I'm beginning to think that eigen classes should not have a solver interface

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



# def eigensolve(self, omega, Ky, num_modes = 10, sigma = 0):
#     '''
#         solve for the kx eigenvalue for a given omega, Ky
#         we can do this by incorporating it into a bloch boundary
#     '''
#     ## add in omega:
#     ## if we want to have 2 dimensions? then what?
#     #omega = 2*np.pi*C0/wvlen;
#     I = sp.identity(self.M); #identity matrix
#     K_omega  = self.Kop - omega**2*MU0*EPSILON0*I;
#     K_omega_ky = K_omega - self.invTepzz@(1j*((self.grid.Dyf + self.grid.Dyb))*Ky + Ky**2*I)
#     OB = sp.bmat([[self.Mop, None],[None, I]]);
#     OA = sp.bmat([[self.Cop, K_omega],[-I, None]]);
#     self.OA = OA;
#     self.OB = OB;
#
#     # solve generalized eigenvalue problem instead
#     eigenvals, eigenmodes = sp.linalg.eigs(self.OA, M = self.OB, k=num_modes, sigma = sigma)
#
#     # D = bslash(self.OB, self.OA);
#     # eigenvals, eigenmodes = sp.linalg.eigs(D, k=num_modes, sigma = sigma)
#
#     return eigenvals, eigenmodes;
