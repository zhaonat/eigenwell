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
    def __init__(self, structure):
        '''
            structure should be of class structure
        '''
        
        self.structure = structure;
        self.grid = FiniteDifferenceGrid(structure.dL,structure.N);

    @abstractmethod
    def make_operator_components():
        pass
