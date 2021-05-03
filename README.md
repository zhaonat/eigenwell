# eigenwell
An objective-oriented approach to finite difference based eigensolving. Eigensolving is more nuanced than it initially might seem. What eigenvalue are you solving for, frequency or k? How do you formulate the problem for each? FDFD offers an interesting point of flexibility in eigenvalue solving 

Plane wave expansion is based on breaking the problem into the Fourier domain. As a result, you cannot solve for the eigenvalue of k given $\omega$

There are some references including analytic solutions to some example problems and some simple pwe code in 1D in the notebooks section as well for comparison.

### Conventions
The code in eigenwell implements everything using

## omega eigenproblem
In this type of problem, there is no resolution of the wavevector k, we only solve for the eigenvalue $\omega$. For example, the modes of a waveguide.

### Classic Dielectric Waveguide

## omega to k eigenproblem
In this problem, we can specify real $\omega$ and get all possible k's, including complex k's when we are, for example, in a bandgap.

### 1D Bragg Mirror
In this eigenproblem, we are looking modes perpendicular to the surface of the 1D Bragg Mirror

![Alt text](./img/bragg_mirror.png?raw=true "Title")


### insulator-metal-insulator example
<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1"> add text

![Alt text](./img/IMI_band_structure.png?raw=true "Title")

### 2D Photonic Crystal

## 3D Waveguide Problem
![Alt text](./img/conductor_3D_waveguide.png?raw=true "Title")


# Examples
these contain python scripts meant to be run from the command line (vs the Jupyter notebooks). Among these are parallel implementations for obtaining bandsolvers using python's multiprocessing module

## Current Problems
1. Implementing PMLs and PECs in a universal way across all eigensolvers.
