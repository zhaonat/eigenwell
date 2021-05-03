from .eigen import *
from .constants import *
import scipy.sparse.linalg as la

class EigenGuideK2D(Eigen):
    '''
        variation in x, wavevector solved in y, i.e. ky
    '''
    def __init__(self, eps_r, grid, polarization = 'TE'):
        super().__init__(eps_r, grid);
        self.polarization = polarization
        self.make_operator_components();

    def make_operator_components(self):

        Epxx = self.grid_average(EPSILON0*self.eps_r.flatten(), 'x');
        invTepxx = sp.spdiags(1/(Epxx), 0, self.M, self.M)

        self.Tepzz = sp.spdiags(EPSILON0*self.eps_r.flatten(), 0,  self.M, self.M)
        self.Tepxx = sp.spdiags(Epxx, 0,  self.M, self.M)

        if(self.polarization == 'TM'):
            A = self.Tepzz@self.grid.Dxb@(invTepxx)@self.grid.Dxf;
        elif(self.polarization == 'TE'):
            A = self.grid.Dxf @ self.grid.Dxb;
        ## how abou tthe TE polarization?
        A = A.astype('complex')
        self.A = A;

    def update_structure(self, eps_r):
        '''
            use this to do dispersive eigensolves
        '''
        self.eps_r = eps_r;
        self.make_operator_components();

    def eigensolve(self, omega, sigma = 0, num_modes = 10):
        '''
            solve for the k eigenvalue for a given omega
        '''
        I = sp.identity(self.M); #identity matrix
        if(self.polarization == 'TM'):
            A = self.A + self.Tepzz*(omega**2*MU0)
        elif(self.polarization == 'TE'):
            A = self.A + omega**2*MU0*self.Tepzz;

        ksqr, modes = la.eigs(A, k=num_modes, sigma = sigma)
        return ksqr, modes;


class EigenGuide3D(Eigen):
    '''
        eps_r:
        grid: grid object storing the derivative operators
    '''

    def __init__(self, eps_r, grid):

        super().__init__(structure, grid);
        self.make_operator_components();
        return;

    def eigensolve(sigma = 0, num_modes = 10):
        ksqr, modes = la.eigs(self.A, k=num_modes, sigma = sigma)
        return ksqr, modes;

    def make_operator_components():
        ## generate operator

        epxx= grid_average(epsilon,'x')
        epyy = grid_average(epsilon, 'y')

        Tez = sp.diags(EPSILON0*epsilon.flatten(), 0, (M,M))
        Tey = sp.diags(EPSILON0*epyy.flatten(), 0, (M,M))
        Tex = sp.diags(EPSILON0*epxx.flatten(), 0,(M,M))
        invTez = sp.diags(1/(EPSILON0*epsilon.flatten()), 0, (M,M))

        Dop1 = sp.bmat([[-sel.grid.Dyf], [self.grid.Dxf]])
        print(Dop1.shape)

        Dop2 = sp.bmat([[-self.grid.Dyb,self.grid.Dxb]])

        Dop3 = sp.bmat([[self.grid.Dxb], [self.grid.Dyb]])

        Dop4 = sp.bmat([[self.grid.Dxf,self.grid.Dyf]])

        Tep = sp.block_diag((Tey, Tex))


        A = omega**2*MU0*Tep + Tep@(Dop1)@invTez@(Dop2) + Dop3@Dop4
        self.A = A;
