import numpy as np

class Structure():
    def __init__(self,eps_r, L):
        '''
            eps_r: NxM
            L: 2x1 Lx,Ly array
        '''
        self.N = eps_r.shape
        self.M = np.prod(self.N);
        self.eps_r = eps_r; #ep_zz
        self.L = np.array(L)
        self.dL = self.L/self.N;
        self.xrange = [-self.L[0]/2, self.L[0]/2]
        self.yrange = [-self.L[1]/2, self.L[1]/2]
        self.epxx = self.grid_average(self.eps_r, 'x');
        self.epyy = self.grid_average(self.eps_r, 'y');

    def grid_average(self, center_array, w):
        '''
            sdf;
        '''
        # computes values at cell edges

        xy = {'x': 0, 'y': 1}
        center_shifted = np.roll(center_array, 1, axis=xy[w])
        avg_array = (center_shifted+center_array)/2
        return avg_array
