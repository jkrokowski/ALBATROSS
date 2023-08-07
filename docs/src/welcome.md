# Welcome to FROOT_BAT

**F**EniCSx **R**epresenation **O**f slender **O**bjects for  **T**hree-Dimensional **B**eam **A**nalysis **T**asks (FROOT_BAT) is a package that performs both 2D and 1D analysis for beams with general cross-sections. The package has the following modules:

* Geoemtry pre-processing:
* Cross-sectional Analysis via polynomial expansion for displacements via BAT-CAVE (**C**ross-Sectional **A**nalysis from **V**ariable **E**pansions) 
* Topologically 1D beam models in 3D ambient space using FEniCSx via BAT-WIING (**W**ith **I**ntervals **I**nvolving **N**onlinear **G**eometry)  
* Recovery relationship and post-processing

# Cite us
```none
@article{lsdo2023,
        Author = { Author 1, Author 2, and Author 3},
        Journal = {Name of the Journal},
        Title = {Title of your paper},
        pages = {203},
        year = {2023},
        issn = {0123-4567},
        doi = {https://doi.org/}
        }
```

<!-- Remove/add custom pages from/to toc as per your package's requirement -->

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/background
src/tutorials
src/custom_1
src/custom_2
src/examples
src/api
```
