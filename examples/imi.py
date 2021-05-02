import sys
import matplotlib.pyplot as plt
from .eigen_k import *
from .grid import *



omega_cutoff = 0.83020*omega_p;
wvlen_cutoff = 2*np.pi*c0/omega_cutoff/1e-6;
wvlen_cutoff2 = 2*np.pi*c0/(0.92*omega_p)/1e-6;
lambda_p = 2*np.pi*c0/omega_p/1e-6;

epsilon_diel = 16;

a = 0.2; #lattice constant
Nx = 500
eps_r = epsilon_diel*np.ones((Nx, ))
eps_r = eps_r.astype('complex')
fill_factor = 0.2;
dx= a/Nx;

## instantiate grid
N = [Nx,1];
dL = [dx, 0];
grid = FiniteDifference(dL,N);
## instantiate eigen_1d;
solver = eigenk1D(eps_r, grid, mode = 'TM');

## ======================================================================#

wvlen_scan = np.linspace(0.7,10, 1000);

kspectra = list();
for wvlen in wvlen_scan:
    omega = 2*np.pi*c0/wvlen/L0;
    epsilon_metal = 1-omega_p**2/(omega**2 - 1j*(gamma*omega))
    eps_r[int(Nx/2-fill_factor*Nx/2): int(Nx/2+fill_factor*Nx/2)] = epsilon_metal;
    ## get eigenvals;
    kvals, modes = solver.solve(omega);
    kspectra.append(kvals);

kspectra = np.array(kspectra);
