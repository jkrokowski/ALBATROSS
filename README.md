# ALBATROSS
**A**nalysis **L**ibrary for **B**eams **A**nd **T**hree-dimensional **R**epresentations **O**f **S**lender **S**tructures

A library for modeling slender structures in 3D ambient space that captures complicated cross-sectional behavior (e.g. warping) due to geometric and material effects.

Current models include:
* Static Timoshenko (shear-deformable) 1D beam theory
* Cross-sectional analysis for computing the cross-section properties (e.g. the beam consitutive matrix) of anisotropic and arbitrary geometry cross-sections used in:
  *  Timoshenko based theory (e.g. 6x6 matrix)
  *  Euler-Bernoulli based theory (e.g. 4x4 matrix)


1D analysis code is heavily borrowed from the formulation and wonderful example files from Jeremy Bleyer here:
https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html

## Installation:
The main requirements are FEniCSx, numpy, scipy, pyvista, meshio, gmsh, and sparseqr. These can all be installed following the instructions below.

<!-- For cross-sectional analysis, a python wrapped implementation of a sparse QR decomposition is used which can be installed from PyPi -->

Usage of a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) enviroment is recommended. For faster environment solving, it is highly recommended to follow [these instructions](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) to configure conda to using the  [mamba](https://mamba.readthedocs.io/en/latest/index.html) solver. See this link for configuring 

<!-- First, install the package dependencies. Then clone the repository and pip install: -->

1. Install [FEniCS](https://fenicsproject.org/download/), [pyvista](https://docs.pyvista.org/version/stable/), [SciPy](https://scipy.org/), [meshio](https://github.com/nschloe/meshio) using conda:
   1. ```connda install -c conda-forge fenics-dolfinx mpich pyvista scipy meshio ```
<!-- 2. Install [gmsh](https://gmsh.info/) using ```pip install gmsh``` (conda installing gmsh does not provide the python API) -->
2. Install [sparseqr](https://github.com/yig/PySPQR) , specifically the latest branch on github as the current pypi version will not work. Note that you need to have installed SuiteSparse and ffi on your system. sparseqr can be installed on linux (or WSL) with the following command:
   1. ```sudo apt-get install libsuitesparse-dev)```
   2. ```sudo apt install libffi-dev```
   3. ```git clone https://github.com/yig/PySPQR.git``` 
   4. ```pip install .``` in the top level PySPQR directory.
3. run ```git clone https://github.com/jkrokowski/ALBATROSS.git```
4. From the top level ALBATROSS directory, run ```pip install .```