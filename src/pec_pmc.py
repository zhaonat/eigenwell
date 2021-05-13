'''
  implement as operators
  if you're looking for the pml, the stretched coordinate pml is a preconditioner
'''
from .constants import *
import numpy as np
import scipy.sparse as sp


class PEC_PMC():
    def __init__(self,N):
        self.N = N
        self.generate_mask();
        return;

    def set_custom_mask(self, mask):
        '''
            uses your own mask
        '''
        M = np.prod(mask.shape)
        MxMy = np.reshape(mask, (M,),order = 'F')
        self.mask = sp.diags(MxMy, 0);

    def generate_mask(self):

        xn = list(range(self.N[0]));
        yn = list(range(self.N[1]));

        ## ordering should be 'F' contiguous
        [Xn,Yn] = np.meshgrid(xn,yn, indexing = 'ij');
        M = np.prod(self.N)
        maskx = np.ones(self.N);
        maskx[Xn == 0] = 0;
        maskx[Xn == self.N[0]-1] =0;

        masky = np.ones(self.N);
        masky[Yn == 0] = 0;
        masky[Yn == self.N[1]-1] =0;

        mx = np.reshape(maskx, (M,),order = 'F')
        my = np.reshape(masky, (M,),order = 'F')
        self.mask_x = sp.diags(mx, 0);
        self.mask_y = sp.diags(my, 0);
        self.mask = self.mask_x@self.mask_y;
