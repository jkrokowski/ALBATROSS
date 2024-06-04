# Welcome to ALBATROSS

**A**nalysis **L**ibrary for **B**eams **A**nd **T**hree-dimensional **R**epresentations **O**f **S**lender **S**tructures

This package is a collection of various beam utilities using [FEniCSx](https://fenicsproject.org/). 

Currently supported tasks:

* Cross-sectional analysis of generic 2D cross-sections
* Static analysis beams with variable and arbitrary cross-sections
* Displacement and stress recovery at any defined cross-section

When cross-sectional analysis is paired with an appropriate 1D model, stress and displacement results computational work can be reduced my multiple orders of magnitude. 

Below is are post processed results for a cantilever beam with an applied tip load.

![figure](../docs/src/images/cantilever_beam.png)

A detailed view of the von_mises stress plotted over the cross-section at the fixed end of the beam is shown below:

![figure](../docs/src/images/von_mises.png)

<!-- # Cite us
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
``` -->

<!-- Remove/add custom pages from/to toc as per your package's requirement -->

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/background
src/tutorials
<!-- src/custom_1 -->
<!-- src/custom_2 -->
src/examples
src/api
```
