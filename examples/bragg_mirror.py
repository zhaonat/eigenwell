import numpy as np
import matplotlib.pyplot as plt
import sys
from eigenwell.src import grid, eigen_k, eigen_w


Nx = 100;
N = [Nx,1]
eps_r = np.ones(N)
eps_r[20:80] = 12;


x = np.linspace(-1/2, 1/2, Nx);
a = 1;
wvlen= 1;
c0 = 3e8;
#eps_r = 1+np.sin(2*np.pi*x/a);
dx = a/Nx;
dL = [dx,0]

fd = grid.FiniteDifference(dL,N)
#print(dir(fd))
#print(fd.Dxf)

omega = 2*np.pi*c0/wvlen

solver = eigen_w.EigenOmega1D(eps_r, fd)
