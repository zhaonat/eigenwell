from abc import ABC, abstractmethod

class Structure:
    '''
        input is an array with pixel-by-pixel specs of the structure
        We do all of our epsilon averaging here
    '''
    def __init__(self,eps_r):
        self.eps_r = eps_r;


class Real_Space(Structure):
    def __init__(self,eps_r):
        self.N = eps_r.shape
        self.eps_r = eps_r;

    def grid_average(self, center_array, w):
        '''
            sdf;
        '''
        # computes values at cell edges

        xy = {'x': 0, 'y': 1}
        center_shifted = np.roll(center_array, 1, axis=xy[w])
        avg_array = (center_shifted+center_array)/2
        return avg_array

#class pwestructure(structure):
