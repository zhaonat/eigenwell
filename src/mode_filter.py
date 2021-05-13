'''
    filter out spurious eigenmodes
'''
class Filter:

    def __init__(self,N, structure_mask, pml_mask = None, threshold = 0.5):
        self.structure_mask = structure_mask;
        self.pml_mask = pml_mask;
        self.threshold = threshold;
        return;

    def mode_filter(self,field):
        '''
            filter by absolute field magnitude
        '''
        field_inside = np.sum(np.abs(field[self.structure_mask == 1])**2);
        field_outside = np.sum(np.abs(field[self.structure_mask == 0])**2);
        total_field= np.sum(np.abs(field)**2);
        if(field_inside/total_field<self.threshold):
            return False
        return True;

    def filter_eigen_matrix(self, eigen_matrix, N):
        '''
            eigen_matrix, Nxv where v is number of eigenvalues
        '''
        _, neigs = eigen_matrix.shape;
        eigensols = list();
        for i in range(neigs):
            if(self.mode_filter(eigen_matrix[:,i].reshape(N))):
                eigensols.append(eigen_matrix[:,i])
        return eigensols
