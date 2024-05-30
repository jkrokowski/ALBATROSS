---
title: Background
---

Beam models are useful for simplifying the computation of displacements and stresses of a slender structure. A slender structure is one that has one characteristic dimension that is much greater than the others.

```{figure} ./images/analysis_splitting.png
---
class: with-border
---
Using a beam model to compute displacements of a 3D model
```

A 3D structure is split into a set of *topologically* two dimensional **"cross-sections"** and a *topologically* one dimensional **"beam axis"**. Both the cross-sections and the beam axis are, in general, embedded in three dimensional space.  

The 1D model utilizes the 2D analysis as an input as a type of material property that represents the material and the geometric shape of the beam. Loads and boundary conditions are applied to the 1D model. The 1D analysis computes the displacements and the reaction force & reaction moment distributions along the beam axis. 

Finally, using both the 2D analysis results, the displacement and stresses of the original structure can be recovered. 

## Cross-Sectional Analysis
The cross-sectional analysis involves computing a set of parameters that are are used to relate strains along the beam axis to the reactions

## 1D Analysis


## The Relationship between Cross-Sectional Analysis Outputs and the 1D Analysis

## Recovery Relationships

```