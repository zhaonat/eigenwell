from .eigen import *
from .constants import *
'''
    eigenomega is different because we solve for omega
'''
class EigenOmega1D(Eigen):
    '''
        omega is the solved eigenvalue
        no resolution on K
        eigenvalue problem is not quadratic so we're good
    '''
    def __init__(self, eps_r, grid, mode = 'TE'):

        super().__init__(eps_r, grid);
        self.polarization = mode;
        self.make_operator_components(mode);

    def make_operator_components(self, mode):

        def grid_average(center_array, w):
            '''
                center_array, 1d eps_r
            '''
            # computes values at cell edges

            xy = {'x': 0, 'y': 1}
            center_shifted = np.roll(center_array, 1, axis=xy[w])
            avg_array = (center_shifted+center_array)/2
            return avg_array
        Epxx = grid_average(self.eps_r, 'x');

        invTepxx = sp.spdiags(1/(EPSILON0*self.eps_r.flatten()), 0, self.M, self.M)
        Tepzz = sp.spdiags(EPSILON0*self.eps_r.flatten(), 0, self.M, self.M)
        ## ================================================================
        #A = Tepzz@Dxf@(invTepxx)@Dxb + Tepzz@sp.spdiags(omega**2*mu0*np.ones((Nx,)), 0, Nx,Nx, format = matrix_format);
        ## ============================================================
        # A = Dxf @ Dxb + omega ** 2 * mu0*Tepzz
        # A = A.astype('complex')
        if(mode == 'TE'):
            A = -(1/MU0)*invTepxx@self.grid.Dxf@self.grid.Dxb
        else:
            A = self.grid.Dxf@invTepxx@self.grid.Dxb
        self.A = A;

    def eigensolve(self, num_modes = 10, sigma = 0):
        '''
            solve for the k eigenvalue for a given omega
        '''
        ## get eigenvalues
        eigenvals, eigenmodes = sp.linalg.eigs(self.A, k=num_modes, which = 'SM')

        return eigenvals, eigenmodes;


class EigenOmega2D(Eigen):
    def __init__(self, eps_r, grid, mode = 'TE'):

        super().__init__(eps_r, grid);
        self.polarization = mode;
        self.make_operator_components(mode);

    def make_operator_components(self, mode):

        def grid_average(center_array, w):
            '''
                center_array, 1d eps_r
            '''
            # computes values at cell edges

            xy = {'x': 0, 'y': 1}
            center_shifted = np.roll(center_array, 1, axis=xy[w])
            avg_array = (center_shifted+center_array)/2
            return avg_array
        Epxx = grid_average(self.eps_r, 'x');
        invTepxx = sp.spdiags(1/(EPSILON0*self.eps_r.flatten()), 0, self.M, self.M)
        Tepzz = sp.spdiags(EPSILON0*self.eps_r.flatten(), 0, self.M, self.M)

        if(mode == 'TE'):
            A = -(1/MU0)*invTepxx@(self.grid.Dxf@self.grid.Dxb+ self.grid.Dyf@self.grid.Dyb)
        else:
            A = self.grid.Dxf@invTepxx@self.grid.Dxb
        self.A = A;

    def eigensolve(self, num_modes = 10, sigma = 0):
        '''
            solve for the k eigenvalue for a given omega
        '''
        ## get eigenvalues
        eigenvals, eigenmodes = sp.linalg.eigs(self.A, k=num_modes, which = 'SM')

        return eigenvals, eigenmodes;
