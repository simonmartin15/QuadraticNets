# QuadraticNets

Code accompanying the paper:

**High-Dimensional Analysis of Gradient Flow for Extensive-Width Quadratic Neural Networks**

This repository contains the simulation code used to reproduce all figures of the paper.  It is designed for reproducibility rather than as a general-purpose framework. 

All figures can be generated directly from precomputed results, or recomputed figure-by-figure if desired.

---
## Requirements

Python ≥ 3.9
Modules: torch, matplotlib, numpy, scipy, mpmath, time, pickle, os, sys, argparse

---
## Repository structure

`code/` Core models and utilities  
`model.py` Gradient descent simulations  
`modelHD.py` High-dimensional simulations  
`utils.py` Helper functions

`HDSimulations/` Notebook to generate high-dimensional (HD) simulation data  

`GDSimulations/`  
`run/` Scripts to launch gradient descent (GD) simulations
`postprocess/` Notebook to aggregate GD outputs

`Figures/` Notebook that generates all paper figures

---
## Pipelines overview

There are two independent pipelines:
1. High-dimensional simulations (HD)
2. Gradient descent simulations (GD)

Both ultimately produce data that is loaded by `Figures/figures.ipynb`.
All experiments can be run **figure by figure**. 

---

## High-dimensional simulations (HD)

Used for theoretical/high-dimensional experiments. 
- Lightweight
- Runs on a standard laptop
### Steps
1. Run: `HDSimulations/HighDimSimulations.ipynb`
2. Results are saved automatically in `Simulators/`
3. `Figures/figures.ipynb` loads these results to generate plots

Precomputed results are already provided, so this step can be skipped.

---

## Gradient descent simulations (GD)

Used for finite-dimensional numerical experiments.
- More computationally expensive
- Typically requires GPU or external compute for full runs
### 1. Run simulations
1. For each figure, navigate to: `GDSimulations/run/fig*/` Each folder contains one or two scripts: `run_fig*.py`
2. The number of runs is indicated as a comment inside each file: `#number of runs = N`
3. Run the script **N times** with different indices:

Example:
for i in {0..14}; do  
python run_figX.py --idx=$i  
done

Each run saves outputs to: `Simulators/`
### 2. Postprocess
Aggregate the runs by opening: `GDSimulations/postprocess/reduce.ipynb`
This notebook:
- loads all runs
- computes statistics
- saves reduced data for plotting
Preprocessed files are already included.
### 3. Plot figures
Finally, generate figures with: `Figures/figures.ipynb`
This notebook directly loads the postprocessed data.

---
## Typical usage

- **Only plot the paper figures**: `Figures/figures.ipynb`
- **Recompute HD simulations**: `HDSimulations/HighDimSimulations.ipynb`
- **Recompute GD simulations**:
	1. Run scripts in `GDSimulations/run/fig*/`
	2. Run `GDSimulations/postprocess/reduce.ipynb`
	3. Run `Figures/figures.ipynb`








