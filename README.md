# learnsolnmap

A research codebase for "Learning Hamiltonian flow maps for long-time simulation from numerical-scheme residuals and examples".

## Overview

The project covers the full pipeline:

1. **Data generation** – Julia scripts simulate ground-truth trajectories and training data for various dynamical systems.
2. **Model training** – PyTorch models trained on the generated data using a Hydra-configured experiment system.
3. **Evaluation & analysis** – Jupyter notebooks and utility scripts assess flow-map accuracy, energy drift, and long-time stability.


## Environments & Packages

### Python

`torch`, `pytorch-lightning`, `hydra-core`, `omegaconf`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `plotly`, `wandb`, `matlabengine`

### Julia

`DifferentialEquations`, `StaticArrays`, `DataFrames`, `CSV`, `Distributions`, `MultiFloats`, `ArgParse`, `ProgressMeter`, `LoggingExtras`


## Repository Structure

```
learnsolnmap/
│
├── data_generation/               # Julia scripts for generate training data
│   ├── bash_scripts/
│   │   └── generate_data.sh       # Shell scripts to run data generation
│   ├── configs/                   # Per-problem generation configs
│   │   ├── nco/
│   │   ├── fpu/
│   │   ├── 3body/
│   │   └── ...
│   └── src/
│       ├── problems/              # Problem definitions (nco.jl, fpu.jl, ...)
│       ├── data_generation/       # Core data sampling algorithm (HMC.jl, ...)
│       └── utils/
│
├── deep_learning/                 # PyTorch training codebase
│   ├── bash_scripts/              # Shell scripts to launch training runs
│   │   ├── nco.sh
│   │   ├── fpu.sh
│   │   └── ...
│   ├── configs/                   # Hydra config tree
│   │   ├── train.yaml             
│   │   ├── eval.yaml             
│   │   ├── experiment/            
│   │   ├── module/                
│   │   └── ...
│   ├
│   └── src/
│       ├── train.py               
│       ├── eval.py               
│       ├── networks/              # Network architectures
│       │   ├── sympnet.py         
│       │   ├── resnet.py          
│       │   ├── henon.py           
│       │   └── ...
│       ├── modules/               # Flow-map models (LightningModules)
│       │   ├── solnmap.py         
│       │   ├── variable_timestep.py
│       │   ├── fixed_timestep.py
│       │   └── ...
│       ├── problems/              # Problem definitions
│       │   ├── nco.py             
│       │   ├── fpu.py  
│       │   └── ...
│       ├── datamodules/           
│       ├── integrators/           # Numerical integrators in PyTorch
│       │   ├── symplectic.py
│       │   └── standard.py
│       ├── losses/                
│       ├── callbacks/             
│       └── utils/                 
│
└── notebooks/                     # Jupyter notebooks for analysis & exploration
    ├── nco/                       
    ├── fpu/                                        
    ├── alphaparticle/
    └── notebook_utils/            

```


## Dynamical Systems Covered

| System | Description |
|---|---|
| **NPCO** | Nearly-Periodic Coupled Oscillators |
| **FPUT** | Fermi-Pasta-Ulam-Tsingou Problem |
| **Alpha-Particle** | Alpha-Particle Dynamics in Stellarators |
| **3-Body / N-Body** | Gravitational N-body Problem |

