# eigenwell
finite difference based eigensolving. Eigensolving is more nuanced than it initially might seem. What eigenvalue are you solving for, frequency or k? How do you formulate the problem for each? FDFD offers an interesting point of flexibility in eigenvalue solving 

Plane wave expansion is based on breaking the problem into the Fourier domain. As a result, you cannot solve for the eigenvalue of k given $\omega$

### Conventions
The code in eigenwell implements everything using

## omega eigenproblem

## omega to k eigenproblem

### insulator-metal-insulator example
<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1"> add text

![Alt text](./img/IMI_band_structure.png?raw=true "Title")

### 2D Photonic Crystal

## 3D Waveguide Problem

# Examples
these contain python scripts meant to be run from the command line (vs the Jupyter notebooks). Among these are parallel implementations for obtaining bandsolvers using python's multiprocessing module

# Notebooks
A more visual and intuitive introduction to the functionalities of the FD eigensolves

