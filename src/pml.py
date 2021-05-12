from .constants import *

import numpy as np
import scipy.sparse as sp



class PML:
    '''
        the PML is actually a preconditioner, not a boundary BoundaryCondition
    '''
    def __init__(self, N, Npml, omega):
        self.lnR = -16; #-12;
        self.m = 4; #3.5;
        self.sigma_max = -(self.m+1)*self.lnR/(2*ETA0);
        self.N = np.array(N);
        self.Npml = np.array(Npml);
        self.omega = omega;

    def create_sfactor(self, wrange, s, dirind = 0):
        '''
        also allow only single directional  sfactors as well
        EPSILON0: vacuum permittivity
        mu0: vacuum permeability
        wrange: 1d real space range
        s: 'f' or 'b'
        N: total grid points including PML should be 2*Npml + N0
        Npml: just the size of the Npml
        '''
        assert s in ['f','b'], "s must be char f or b (forward or backward)"
        if(np.sum(self.Npml) == 0):
            return np.ones(self.N[dirind], dtype = 'complex');
        w_array = np.linspace(wrange[0], wrange[1], self.N[dirind]+1);
        #indexes the real space coordinate where the PML begins
        sfactor_array = np.ones(self.N[dirind], dtype = 'complex');

        loc_pml = np.array([w_array[self.Npml[dirind]], w_array[self.N[dirind]-self.Npml[dirind]]]); #specifies where the pml begins on each side
        d_pml = np.abs(np.array(wrange) - loc_pml);  #pml thickness

        ## how to handle divide by 0...
        sigma_max = self.sigma_max / d_pml; #usually the pml is the same thickness on both sides

        if(s == 'b'):
            ws = w_array[:-1]; #truncating size by one...
        elif(s == 'f'):   #s == 'f'
            ws = (w_array[:-1] + w_array[1:]) / 2;

        ind_pml = [ws < loc_pml[0], ws > loc_pml[1]];  #should be two arrays of booleans of size Npml

        for n in [0,1]:
            sfactor = lambda L : 1 - 1j * sigma_max[n]/(self.omega*EPSILON0) * (L/d_pml[n])**self.m;  ## inline function
            r= np.abs(loc_pml[n] - ws[ind_pml[n]])
            sfactor_array[ind_pml[n]] = sfactor(r);
        return sfactor_array;


    def createSws(self, direction, s, wrange):
        '''
            direction: 'x' or 'y'
            s: 'f' or 'b'
            wrange: [-l, l]
        '''
        assert(direction in ['x','y'], "direction must be 'x' or 'y'")
        M = np.prod(self.N)
        Sw_s_2D = np.zeros(self.N, dtype = 'complex');

        if(direction == 'x'):
            sws_vector = self.create_sfactor(wrange,s, dirind = 0);
            for j in range(self.N[1]):
                Sw_s_2D[:, j] = 1/sws_vector;
        elif(direction == 'y'):
            sws_vector = self.create_sfactor(wrange,s, dirind = 1);
            for i in range(self.N[0]):
                Sw_s_2D[i, :] = 1/sws_vector;

        ## we implement order = 'F'
        Sw_s_2D = np.reshape(Sw_s_2D, (M, ), order = 'F')

        Sws=sp.spdiags(Sw_s_2D,0,M,M);
        return Sws, sws_vector

    def Soperators(self, xrange, yrange):
        self.Sxf, self.sxf  = self.createSws('x', 'f', xrange);
        self.Syf, self.syf  = self.createSws('y', 'f', yrange);
        self.Sxb, self.sxb  = self.createSws('x', 'b', xrange);
        self.Syb, self.syb  = self.createSws('y', 'b', yrange);
        return self.Sxf, self.Syf, self.Sxb, self.Syb

class SymmetrizePML(PML):
    '''
        generates left and right symmetrization of the pml
    '''
    def __init__(self,N, Npml, omega, polarization = 'TE'):
        self.polarization = polarization;
        super().__init__(N,Npml,omega);

    def Soperators(self, xrange, yrange):
        '''
            generate Soperators with the symmetrizer...
        '''
        M = np.prod(self.N)
        self.Sxf, _  = self.createSws('x', 'f', xrange);
        self.Syf, _  = self.createSws('y', 'f', yrange);
        self.Sxb, _  = self.createSws('x', 'b', xrange);
        self.Syb, _  = self.createSws('y', 'b', yrange);

        Sxfd = self.Sxf.diagonal();
        Sxbd = self.Sxb.diagonal();
        Syfd = self.Syf.diagonal();
        Sybd = self.Syb.diagonal();

        if(self.polarization == 'TM'):
            numerator = np.sqrt(Sxfd*Syfd);
        elif(self.polarization == 'TE'):
            numerator = np.sqrt(Sxbd*Sybd);

        denominator = 1/(numerator);
        Pr = sp.spdiags(numerator, 0, M,M);
        Pl = sp.spdiags(denominator,0,M,M);

        self.Pl = Pl;
        self.Pr = Pr;
        return Pl, Pr;
