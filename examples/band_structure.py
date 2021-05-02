import numpy as np
import matplotlib.pyplot as plt
import sys
print(sys.path)
from eigenwell.src import grid, eigen_k, eigen_w
from eigenwell.src.constants import *
import multiprocessing

'''
    use multiprocessing to enhance speed
'''

def compute_spectra():
    return;

if __name__ == '__main__':

    Nx = 80;
    Ny = 80;
    N = np.array([Nx, Ny]);

    eps_r = np.ones(N);

    a = np.array([1,1]);
    dL = a/N;
    radius = 0.25;
    ## put in a circle;
    ci = int(Nx/2); cj= int(Ny/2);

    cr = (radius/a[0])*Nx;
    I,J=np.meshgrid(np.arange(eps_r.shape[0]),np.arange(eps_r.shape[1]));

    print(eps_r.shape)
    dist = np.sqrt((I-ci)**2 + (J-cj)**2);
    #print(np.where(dist<cr))
    eps_r[np.where(dist<cr)] = 6;

    plt.imshow(eps_r)
    wvlen_scan = np.logspace(np.log10(0.8), np.log10(10), 600);


    fd = grid.FiniteDifference(dL,N)

    print(fd.Dxf.shape)

    eigk = eigen_k.EigenK2D(eps_r, fd)
    print(fd.Dxf.shape)
    Ky = 0;

    wvlen_scan = np.linspace(1,10,60);
    wvlen_scan = np.logspace(np.log10(1), np.log10(10),200)
    spectra = [];
    for c,wvlen in enumerate(wvlen_scan):
        omega = 2*np.pi*C0/(wvlen);
        eigvals, eigvecs = eigk.eigensolve(omega, Ky, num_modes = 3)
        spectra.append(eigvals);
        if(c%5 == 0):
            print(c, wvlen)
    spectra = np.array(spectra)
