'''
  implement as operators
  if you're looking for the pml, the stretched coordinate pml is a preconditioner
'''
from .constants import *
import numpy as np
import scipy.sparse as sp

class BoundaryCondition:
    '''
        not sure for now whether a parent class is reasonable
    '''
    def __init__(self, wrange, Nw, Nw_pml):
        return;

    def generate_mask():
        pass;

class PEC_PMC(BoundaryCondition):
    def __init__(self,N):
        self.N = N
        self.generate_mask();
        return;

    def generate_mask(self):

        xn = list(range(self.N[0]));
        yn = list(range(self.N[1]));
        [Xn,Yn] = np.meshgrid(xn,yn);

        maskx = np.ones(self.N);
        maskx[Xn == 0] = 0;
        maskx[Xn == self.N[0]-1] =0;

        masky = np.ones(self.N);
        masky[Yn == 0] = 0;
        masky[Yn == self.N[1]-1] =0;

        self.mask_x = sp.diags(maskx.flatten(), 0);
        self.mask_y = sp.diags(masky.flatten(), 0);
