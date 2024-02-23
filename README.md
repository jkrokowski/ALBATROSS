# ALBATROSS
**A**nalysis **L**ibrary for **B**eams **A**nd **T**hree-dimensional **R**epresentations **O**f **S**lender **S**tructures

A library for modeling slender structures implementing topologically 1D beam models in 3D ambient space using FEniCSx includeing cross-sectional analysis for determining section properties.

Current models include:
* Static Timoshenko (shear-deformable) 1D beam theory
* Cross-sectional analysis for computing the cross-section properties of arbitrary shapes

1D analysis code is heavily borrowed from the formulation and wonderful example files from Jeremy Bleyer here:
https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html

## Installation:
The main requirements are FEniCSx, beyond the typical numpy, scipy, etc requirements. 
For cross-sectional analysis, a python wrapped implementation of a sparse QR decomposition is used which can be installed from PyPi
Usage of a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or for even faster installation, [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)) enviroment is recommended.

First, install the package dependencies. Then clone the repository and pip install:

1. Install [Numpy](https://numpy.org/) and [SciPy](https://scipy.org/)
2. Install [FEniCS](https://fenicsproject.org/download/) and [pyvista](https://docs.pyvista.org/version/stable/) for visualization
3. Install [sparseqr](https://github.com/yig/PySPQR)
4. run ```git clone https://github.com/jkrokowski/ALBATROSS.git```
5. From the top level ALBATROSS directory, run ```pip install .```