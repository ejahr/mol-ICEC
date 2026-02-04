# Analytical model for ICEC with intramolecular vibrational motion
Interatomic Coulombic Electron Capture (ICEC) is an environment-mediated process in which a free electron attaches to a species by transferring excess energy to a neighbor. 
This implentation incorporates the vibrational degrees of freedom of the neighbor into an analytical model of ICEC. 
The results are cross sections or electron spectra depending on the incoming electron energy.

## Files
- `input/` : defines the general input parameters.
- `H-LiH.py` : Defines and runs the calculation and plots for the ICEC cross sections. Uses classes and functions defined in `icec/`. Results in [](). Run with `python H-LiH.py`.
- `icec` : contains the classes ICEC and IntraICEC which are the base modules for the calculations.

## Requirements
- Can be run on Linux or WSL. Not tried on native Windows or iOS.
- Python >= 3.11.10
- Libraries: `numpy`, `scipy`, [`mpmath`](https://mpmath.org/), `matplotlib`
