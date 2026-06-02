# Analytical model for ICEC with intramolecular vibrational motion
Interatomic Coulombic Electron Capture (ICEC) is an environment-mediated process in which a free electron attaches to an atom or molecule by transferring excess energy to a neighbor, ionizing said neighbor.

$$e^-_k + \mathrm{A} + \mathrm{D} \to \mathrm{A}^- + \mathrm{D}^+ + e^-_{k'}$$

This implentation incorporates the vibrational degrees of freedom of the neighbor into an analytical model of ICEC. 
The results are cross sections or electron spectra depending on the incoming electron energy.

## Theory
E. M. Jahr, E. Fasshauer; Intramolecular nuclear dynamics in intermolecular Coulombic electron capture. _J. Chem. Phys._ 164, 194304 (2026). https://doi.org/10.1063/5.0333097

The electronic ICEC cross section is given as

$$
\sigma (\varepsilon) = 
\frac{3 c^4}{4 \pi} 
     \frac{\sigma^\text{PR}_\mathrm{A}(\varepsilon) \sigma^\text{PI}_\mathrm{D}(\omega)}
         {R^6 \omega^4}
$$

where $\varepsilon$ is the incoming electron energy corresponding to momentum $k$, $\sigma^\mathrm{PR}$ and $\sigma^\mathrm{PI}$ are the photorecombination and photoionization cross sections, and $R$ is the distance between A and D.
The transferred energy $\omega$ is

$$\omega = \varepsilon + \mathrm{IP}_{\mathrm{A}^-}$$

The photorecombination cross section can be optained from the photoionization cross section via the principle of detailed balance

$$
\sigma^\mathrm{PR}_\mathrm{A}(\varepsilon)
= \frac{\omega^2}{2\varepsilon c^2} \frac{g_{\mathrm{A}^-}}{g_\mathrm{A}}  \sigma^\mathrm{PI}_{\mathrm{A}^-}(\omega)
$$

The ICEC cross section including the vibronic transition $\nu_\mathrm{D} \to \nu_{\mathrm{D}^+}$ of the neighbor is

$$\sigma(\varepsilon, \nu_\mathrm{D} \to \nu_{\mathrm{D}^+}) 
    = \frac{3 c^4 }{4 \pi} 
    \frac{
    \sigma^\mathrm{PR}_\mathrm{A}(\varepsilon) 
    \sigma^{\mathrm{PI}}_{\nu_\mathrm{D} \nu_{\mathrm{D}^+}}(\omega)
    }{\omega^4 R^6}$$
    
If the Franck-Condon approximation is used, the ICEC cross section can be written as

$$
\sigma(\varepsilon, \nu_\mathrm{D} \to \nu_{\mathrm{D}^+}) =
    \frac{3 c^4 }{4 \pi}
    \frac{
    \sigma^\mathrm{PR}_\mathrm{A}(\varepsilon)
    \sigma^\mathrm{PI}_\mathrm{D}(\omega)
    }{\omega^4 R^6}
     |\langle \nu_\mathrm{D} | \nu_{\mathrm{D}^+} \rangle |^2  
$$

The Morse potential to calculate the Franck-Condon factors $|\langle \nu_\mathrm{D} | \nu_{\mathrm{D}^+} \rangle |^2$ is given by

$$V(R) = D_e \left( 1 - e^{\alpha (R-R_\text{eq})}\right)^2 - D_e$$

## Files
- `config.py` : defines the input parameters, currently for H+ LiH
- `run_system.py` : script to start the calculation and plots based on the parameters given in `config.py`.

## Sub packages
- `icec` : contains the classes `Units` and `Constants` (`constants.py`), `ICEC` (`icec.py`), `IntraICEC` and `RydbergIntraICEC` (`IntraICEC.py`), and `Morse` (`morse.py`).
            `Units` defines unit conversion factors, `Constants` defines constants in atomic units.
            `ICEC` and `IntraICEC` forms the basic modules for the calculation of ICEC and ICEC with intramolecular nuclear motion cross sections.
- `plotting`: plotting functions for total cross sections (`cross_section.py`), electron spectra (`spectra.py`), and potential energy surfaces (`pes.py`).
- `calc`: functions for calculating and saving cross sections (`cross_section.py`) and spectra (`spectra.py`), based on `icec`. Includes `fit.py` for fitting photoionization cross sections and `file_io.py` for reading and writing files.

## Example H+ LiH
- `HLiH/config.py` : defines the input parameters for H+ LiH
- `HLiH/H-LiH.py` : script to start the calculation and plots for H+ and LiH.
- `HLiH/print_infos.py`: prints additional information for H+ LiH to the console.
- `HLiH/data/` : photoionization cross sections of H and LiH. 
- `HLiH/results/`: results of the calculations from `H-LiH.py`. Includes `LiHp.diss_energies.E2.L8.txt`.
- `HLiH/plots/`: generated plots which show the calculated ICEC results. Also includes PES of LiH, an energy sketch and a visualization of the dissociative states of LiH+.

## Requirements
- Can be run on Linux or WSL. Not tried on native Windows or iOS.
- Python = 3.12
- Libraries: `numpy`, `scipy`, [`mpmath`](https://mpmath.org/), `matplotlib`, `concurrent.futures`, `itertools` 
