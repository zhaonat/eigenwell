from .constants import *
#import boundary_conditions
from .grid import *
from abc import ABC, abstractmethod

'''
    Abstract base class
'''

class Eigen(ABC):
    '''
        superclass
    '''
    def __init__(self, structure, grid):

        assert(structure.shape[0] == grid.N[0] and structure.shape[1] == grid.N[1] ,\
         "grid and structure should have same shape")

        self.N = structure.shape
        self.M = np.prod(self.N);
        self.eps_r = structure;
        self.grid = grid;
        
    def grid_average(self, center_array, w):
        '''
            center_array, 1d eps_r
        '''
        # computes values at cell edges

        xy = {'x': 0, 'y': 1}
        center_shifted = np.roll(center_array, 1, axis=xy[w])
        avg_array = (center_shifted+center_array)/2
        return avg_array

    @abstractmethod
    def eigensolve():
        pass
    @abstractmethod
    def make_operator_components():
        pass
