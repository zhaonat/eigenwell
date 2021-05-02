from numpy import sqrt, pi

'''
    important units in SI
'''

EPSILON0 = 8.854e-12 ;
MU0 = pi * 4e-7 ;
C0 = 1 / sqrt(EPSILON0 * MU0);
ETA0 = sqrt(MU0/EPSILON0);

matrix_format = 'csc_matrix'; #csr_matrix
