'''
    we have seperate operators for omega to k
    vs k to omega
'''
import numpy as np
import scipy.sparse as sp

class FiniteDifference():
    '''
        works for 1d and 2d since we have TE and TM seperation
        fdfd sets up: 1 the matrix operator A:
        which may differ if it's a k to omega or omega to k solve
        A = dxf*dxb+dyf*dyb+omega^2 (mu_0, eps_r)
    '''
    def __init__(self, dL,N):
        '''
            for 1d, use N = [Nx,1]
            dL: [dx, dy] discretization
            eps_r: pixel-by-pixel Nx x Ny structure of relative permittivities
            omega: frequency
        '''
        assert (len(dL)==len(N) == 2, 'must specify 2 elem arr even for 1d sims')
        assert 0 not in dL and 0 not in N, 'do not make anything 0 in dL or N'
        self.dL = dL;
        self.N = N;
        self.make_derivatives();
        return;

    def createDws(self, s, f):
        '''
            s = 'x' or 'y': x derivative or y derivative
            f = 'b' or 'f'
            catches exceptions if s and f are misspecified
        '''
        M = np.prod(self.N);
        Nx = self.N[0];
        Ny = self.N[1];

        sign = -1 if f == 'f' else 1;
        dw = None; #just an initialization
        indices = np.reshape(np.arange(M), (Ny,Nx));
        if(s == 'x'):
            ind_adj = np.roll(indices, sign, axis = 1)
            dw = self.dL[0]
        elif(s == 'y'):
            ind_adj = np.roll(indices, sign, axis = 0)
            dw = self.dL[1];

        on_inds = np.hstack((indices.flatten(), indices.flatten()))
        off_inds = np.concatenate((indices.flatten(), ind_adj.flatten()), axis = 0);
        all_inds = np.concatenate((np.expand_dims(on_inds, axis =1 ), np.expand_dims(off_inds, axis = 1)), axis = 1)

        data = np.concatenate((-sign*np.ones((M)), sign*np.ones((M))), axis = 0)
        Dws = sp.csc_matrix((data, (all_inds[:,0], all_inds[:,1])), shape = (M,M));

        return (1/dw)*Dws;

    ## overloaded operators (can we have multiple functions with the same name?)
    # python does not natively support overloading
    def make_derivatives(self, PML = None, PEC_PMC = None):
        if(PML):
            self.Dxf = PML.Sxf@self.createDws('x', 'f');
            self.Dyf = PML.Syf@self.createDws('y', 'f');
            self.Dxb = PML.Sxb@self.createDws('x', 'b');
            self.Dyb = PML.Syb@self.createDws('y', 'b');
        if(PEC_PMC):
            return;

        if(not PML and not PEC_PMC):
            self.Dxf = self.createDws('x', 'f');
            self.Dyf = self.createDws('y', 'f');
            self.Dxb = self.createDws('x', 'b');
            self.Dyb = self.createDws('y', 'b');


        return;
