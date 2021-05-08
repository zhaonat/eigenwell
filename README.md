# eigenwell
An objective-oriented approach to finite difference based eigensolving for Maxwell's equations. Eigensolving is more nuanced than it initially might seem. What eigenvalue are you solving for, frequency or k? How do you formulate the problem for each? FDFD offers an interesting point of flexibility in eigenvalue solving 

Plane wave expansion is based on breaking the problem into the Fourier domain. As a result, you cannot solve for the eigenvalue of k given $\omega$

There are some references including analytic solutions to some example problems and some simple pwe code in 1D in the notebooks section as well for comparison.

## Conventions
TM polarization: Hz, Ex, Ey (H field out of plane)
TE polarization: Ez, Hx, Hy (E field out of plane)

1D simulations are along the x axis. 2D simulations are along the x and y axes

## Eigensolving Workflow
Because eigensolving on sparse FDFD matrices can be tricky, the eigen classes do not implement any solvers. Instead, you will have access to the final operator and you will have the responsibility of setting up the sparse eigensolving problem as you desire. typically though, eigensolving requires you to put in a guess of omega

### Eigensolving with PMLs


## omega eigenproblem
In this type of problem, there is no explicit resolution of the wavevector k, we only solve for the eigenvalue $\omega$. For example, the modes of a waveguide. However, we can include fixed Kx or Ky into the grid by using a bloch boundary condition. 

### Classic Dielectric Waveguide
![Alt text](./img/dielectric_guide_mode.png?raw=true "Title")

### Surface Plasmons (TM)
![Alt text](./img/surface_plasmons.png?raw=true "Title")

## omega to k eigenproblem
In this problem, we can specify real $\omega$ and get all possible k's, including complex k's when we are, for example, in a bandgap. This type of problem means we need to extract the exp(ikx) or exp(iky) dependence analytically and modify the form of the operator.

### 1D Bragg Mirror
In this eigenproblem, we are looking modes perpendicular to the surface of the 1D Bragg Mirror

![Alt text](./img/bragg_mirror.png?raw=true "Title")


### insulator-metal-insulator example
<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1"> add text

![Alt text](./img/IMI_band_structure.png?raw=true "Title")

### 2D Photonic Crystal

![Alt text](./img/2d_phc_band_structure.png?raw=true "Title")



## 3D Waveguide Problem

Example below is a metal conductor waveguide.
![Alt text](./img/conductor_3D_waveguide.png?raw=true "Title")


# Examples
these contain python scripts meant to be run from the command line (vs the Jupyter notebooks). Among these are parallel implementations for obtaining bandsolvers using python's multiprocessing module

# Implementation Notes
Note that python uses 'C'-contiguous ordering of its n>1 dimensional arrays. I will be using 'F' ordering of the arrays (which is what MatLab) uses. That means when you reshape a flat eigenvector, you should do np.reshape(flat_array, new_dim, ordering = 'F'), otherwise your result will look messed up.

## Current Problems
1. Implementing PMLs and PECs in a universal way across all eigensolvers. I'd like to implement everything as left and right preconditioners, but that might not be viable so I'm using workarounds at present.
2. correct bloch boundary
