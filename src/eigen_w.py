from .eigen import *
from .constants import *
from .grid import *
from .pml import *
from .pec_pmc import *
'''
    eigenomega is different because we solve for omega
'''

class EigenOmega2D(Eigen):
    '''
        challenge: npml means we need to specify a frequency, which is not great
    '''
    def __init__(self, eps_r, omega_guess = None,
                              npml = [0,0],
                              pec_pmc = False,
                              polarization = 'TE'):
        '''
            npml: pml thickness in number of cells
            omega_guess

            how do we specify a boundary condition?
        '''
        super().__init__(eps_r);
        self.polarization = polarization;
        self.omega = omega_guess;
        self.npml = npml;
        self.pec_pmc = pec_pmc;
        self.make_operator_components();


    def make_operator_components(self):

        M = self.structure.M;
        N = self.structure.N;
        Dxf = self.grid.Dxf; Dyf = self.grid.Dyf;
        Dxb = self.grid.Dxb; Dyb = self.grid.Dyb;

        if(self.omega!=None and np.sum(self.npml)>0):
            pml_obj = PML(self.structure.N, self.npml, self.omega)
            pml_obj.Soperators(self.structure.xrange, self.structure.yrange);

            Sxf, Syf = pml_obj.Sxf, pml_obj.Syf;
            Sxb, Syb = pml_obj.Sxb, pml_obj.Syb;

            Dxf, Dxb = Sxf@Dxf, Sxb@Dxb;
            Dyf, Dyb = Syf@Dyf, Syb@Dyb;

        pec_pmc_mask = sp.identity(M);
        if(self.pec_pmc == True):
            pec_pmc_obj = PEC_PMC(N);
            mask_x = pec_pmc_obj.mask_x
            mask_y = pec_pmc_obj.mask_y;
            pec_pmc_mask = pec_pmc_obj.mask_x@pec_pmc_obj.mask_y;

        Epxx = np.reshape(self.structure.epxx, (M,), order = 'F')
        Epyy = np.reshape(self.structure.epyy, (M,), order = 'F');
        Epzz = np.reshape(self.structure.eps_r, (M,), order = 'F');

        invTepxx = sp.spdiags(1/(EPSILON0*Epxx), 0, M,M)
        invTepyy = sp.spdiags(1/(EPSILON0*Epyy), 0, M,M)
        invTepzz = sp.spdiags(1/(EPSILON0*Epzz), 0, M,M)

        if(self.polarization == 'TE'):
            A = -(1/MU0)*pec_pmc_mask@(invTepzz@(Dxb@Dxf+ Dyb@Dyf))@pec_pmc_mask;
        elif(self.polarization == 'TM'):
            A = -(1/MU0)* pec_pmc_mask@( Dxf@(invTepxx)@Dxb + Dyf@(invTepyy)@Dyb )@pec_pmc_mask;
        self.A = A;
