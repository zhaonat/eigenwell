from .eigen import *
from .constants import *
import scipy.sparse.linalg as la

class EigenGuide2D(Eigen):
    '''
        variation in x, wavevector solved in y, i.e. ky
        eps_r_struct: obj of structure which contains eps_r
        eigenvalue being solved is ky^2 in direction perpendicular to x
        ksqr, modes = la.eigs(A, k=num_modes, sigma = sigma)

    '''
    def __init__(self, eps_r_struct, polarization = 'TE'):
        super().__init__(eps_r_struct);
        self.polarization = polarization
        self.make_operator_components(0);

    def make_operator_components(self, omega):

        M = self.structure.M
        Dxf = self.grid.Dxf; Dxb = self.grid.Dxb;

        Epxx = np.reshape(self.structure.epxx, (M,), order = 'F')
        invTepxx = sp.spdiags(1/(EPSILON0*Epxx), 0, M, M)
        Epzz = np.reshape(self.structure.eps_r, (M,), order = 'F');
        Tepzz = sp.spdiags(EPSILON0*Epzz, 0,  M, M)

        if(self.polarization == 'TM'):
            A = Tepzz@Dxb@(invTepxx)@Dxf +Tepzz*(omega**2*MU0);
        elif(self.polarization == 'TE'):
            A = Dxf @ Dxb + omega**2*MU0*Tepzz;
        ## how abou tthe TE polarization?
        A = A.astype('complex')
        self.A = A;

    def update_structure(self, eps_r_struct):
        '''
            use this to do dispersive eigensolves
        '''
        self.structure = eps_r_struct;


class EigenGuide3D(Eigen):
    '''
        eps_r:
        grid: grid object storing the derivative operators
    '''

    def __init__(self, eps_r, grid):

        super().__init__(eps_r, grid);
        self.make_operator_components();
        return;

    def make_operator_components(self, omega):
        '''
            return a function that is a function of omega?
        '''

        epsilon = self.eps_r;
        epxx= grid_average(epsilon,'x')
        epyy = grid_average(epsilon, 'y')

        Tez = sp.diags(EPSILON0*epsilon.flatten(), 0, (self.M,self.M))
        Tey = sp.diags(EPSILON0*epyy.flatten(), 0,  (self.M,self.M))
        Tex = sp.diags(EPSILON0*epxx.flatten(), 0, (self.M,self.M))

        invTez = sp.diags(1/(EPSILON0*epsilon.flatten()), 0,  (self.M,self.M))

        Dop1 = sp.bmat([[-Dyf], [Dxf]])
        Dop2 = sp.bmat([[-Dyb,Dxb]])
        Dop3 = sp.bmat([[Dxb], [Dyb]])
        Dop4 = sp.bmat([[Dxf,Dyf]])

        Tep = sp.block_diag((Tey, Tex))
        self.A =  Tep@(Dop1)@invTez@(Dop2) + Dop3@Dop4;

    def update_operator(self,omega):

        Tey = sp.diags(EPSILON0*epyy.flatten(), 0,  (self.M,self.M))
        Tex = sp.diags(EPSILON0*epxx.flatten(), 0, (self.M,self.M))
        Tep = sp.block_diag((Tey, Tex))

        return self.A + omega**2*MU0*Tep;
