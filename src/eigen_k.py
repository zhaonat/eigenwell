from .eigen import *
from .constants import *
from scipy.sparse.linalg import spsolve as bslash

class EigenK1D(Eigen):
    '''
        eps_r: numpy array of the structure
        grid: grid object storing the derivative operators
    '''

    def __init__(self, structure, grid, polarization = 'TE'):

        super().__init__(structure, grid);
        self.polarization = polarization
        self.make_operator_components();

        return;

    def make_operator_components(self):

        invTepzz = sp.spdiags(1 / self.eps_r.flatten(), 0, self.M,self.M)
        if(self.polarization == 'TM'):
            self.Mop = invTepzz;
            self.Cop = -invTepzz@(-1j * (self.grid.Dxf + self.grid.Dxb));
            self.Kop = -invTepzz@(self.grid.Dxf @ self.grid.Dxb) #- omega**2*mu0*eps0*I;
        elif(self.polarization == 'TE'):
            self.Mop = invTepzz;
            self.Cop = -invTepzz@(-1j * (self.grid.Dxf + self.grid.Dxb));
            self.Kop = -invTepzz@(self.grid.Dxf @ self.grid.Dxb) #- omega**2*mu0*eps0*I;
        ## how abou tthe TE polarization?

        return;

    def update_structure(self, eps_r):
        self.eps_r = eps_r;
        self.make_operator_components(self.mode);

    def eigensolve(self, omega, sigma = 0, num_modes = 10):
        '''
            solve for the k eigenvalue for a given omega
        '''

        I = sp.identity(self.M); #identity matrix
        invTepzz = sp.spdiags(1 / self.eps_r.flatten(), 0, self.M,self.M)
        K_omega  = self.Kop - omega**2*MU0*EPSILON0*I;

        OB = sp.bmat([[self.Mop, None],[None, I]]);
        OA = sp.bmat([[self.Cop, K_omega],[-I, None]]);
        self.OA = OA;
        self.OB = OB;
        D = bslash(self.OB, self.OA);
        eigenvals, eigenmodes = sp.linalg.eigs(D, k=num_modes, sigma = sigma)
        return eigenvals, eigenmodes;

class EigenK2D(Eigen):
    '''

    '''
    def __init__(self, structure, grid, polarization = 'TE'):

        super().__init__(structure, grid);
        self.polarization = polarization
        self.make_operator_components();

        return;

    def make_operator_components(self):

        Epxx = self.grid_average(self.eps_r, 'x');
        Epyy = self.grid_average(self.eps_r, 'y');

        self.invTepzz = sp.spdiags(1 / self.eps_r.flatten(), 0, self.M,self.M)
        if(self.polarization == 'TM'):
            self.Mop = self.invTepzz;
            self.Cop = -self.invTepzz@(-1j * (self.grid.Dxf + self.grid.Dxb));
            self.Kop = -self.invTepzz@(self.grid.Dxf @ self.grid.Dxb) #- omega**2*mu0*eps0*I;

        elif(self.polarization == 'TE'):
            self.Kop = self.invTepzz@(-self.grid.Dxf @ self.grid.Dxb - self.grid.Dyf @ self.grid.Dyb)# - 1j*((Dyf + Dyb))*Ky + Ky**2*I) ;
            self.Mop = self.invTepzz;
            self.Cop = -self.invTepzz@(1j * (self.grid.Dxf + self.grid.Dxb)); #% lambda


    def update_structure(self, eps_r):
        self.eps_r = eps_r;
        self.make_operator_components(self.mode);

    def eigensolve(self, omega, Ky, num_modes = 10, sigma = 0):
        '''
            solve for the kx eigenvalue for a given omega, Ky
        '''
        ## add in omega:
        ## if we want to have 2 dimensions? then what?
        #omega = 2*np.pi*C0/wvlen;
        I = sp.identity(self.M); #identity matrix
        K_omega  = self.Kop - omega**2*MU0*EPSILON0*I;
        K_omega_ky = K_omega - self.invTepzz@(1j*((self.grid.Dyf + self.grid.Dyb))*Ky + Ky**2*I)
        OB = sp.bmat([[self.Mop, None],[None, I]]);
        OA = sp.bmat([[self.Cop, K_omega],[-I, None]]);
        self.OA = OA;
        self.OB = OB;
        D = bslash(self.OB, self.OA);
        eigenvals, eigenmodes = sp.linalg.eigs(D, k=num_modes, sigma = sigma)

        return eigenvals, eigenmodes;
