# Getting started


## Installation:
The main requirements are FEniCSx, beyond the typical numpy, scipy, etc requirements. 
For cross-sectional analysis, a python wrapped implementation of a sparse QR decomposition is used which can be installed from PyPi. 

Usage of a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or for even faster installation, [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)) enviroment is recommended.

First, install the package dependencies. Then clone the repository and pip install:

1. Install [Numpy](https://numpy.org/) and [SciPy](https://scipy.org/)
2. Install [FEniCS](https://fenicsproject.org/download/) and [pyvista](https://docs.pyvista.org/version/stable/) for visualization
3. Install [sparseqr](https://github.com/yig/PySPQR) from github (not the PyPi package as this is missing key functionality)
4. Install mesh handling package [meshio]https://github.com/nschloe/meshio 
5. Install the python api for [gmsh]https://gmsh.info/ via pip
6. run ```git clone https://github.com/jkrokowski/ALBATROSS.git```
7. From the top level ALBATROSS directory, run ```pip install .```

## Understanding the theory
If you are unfamiliar with the theory behind beam models and their place within the solid mechanics world, we suggest you start at the [background](./background.md) page. 

## Running your first analysis
For the "hello world" example of a beam model, start with the [hello beam tutorial](./tutorials/hello_beam) 