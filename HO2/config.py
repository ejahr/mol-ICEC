''' 
Defines the parameters needed for the calculation of ICEC cross sections and spectra.
run_system.py starts the calculations and plots.
All parameters should be given (or converted to) atomic units.
'''

import os
from icec.constants import Units, Constants

DIR = os.path.dirname(os.path.realpath(__file__))   # DIR where this config.py file is located
DIR_DATA = os.path.join(DIR, 'HO2', 'data')         # dir name where PI XS data is located
DIR_RESULTS = os.path.join(DIR, 'HO2', 'results')   # dir name where results should be saved
DIR_PLOTS = os.path.join(DIR, 'HO2', 'plots')       # dir name where plots should be saved       

# === System ===
system_name = 'Hp-O2'                               # used for file names
reaction    = 'e- + H+ + O2 -> H + (O2)+ + e-'      # used in the header of result files

# === Parameters for the calculations ===
R           = 3.9522557993362915 * Units.ANGSTROM2BOHR   # distance between center of masses of A and D
L           = 8     * Units.ANGSTROM2BOHR   # box length for discretizing the dissociative states of D+

vD_max_FC   = 1                             # maximum initial vibrational state considered for bound-continuum transitions
min_kinE    = 0.01  * Units.EV2HARTREE      # minimum incoming electron energy
max_kinE    = 9     * Units.EV2HARTREE      # maximum incoming electron energy
max_dissE   = 2     * Units.EV2HARTREE      # maximum energy up to which the dissociative states are calculated
num_grid    = 100                          # number of grid points between min and max electron energies
n_max_ryd   = 10                            # number of rydberg states

electronE   = 1     * Units.EV2HARTREE      # incoming electron energy for ICEC electron spectra
T           = [15, 300, 1500]   # Kelvin    # temperatures for temperature dependent spectra

# === Define which calculations are active ===
calc_roots      = 0         # activates calculation of the dissociative states of D+ within box of L, needs to run at least once
resolved        = 0         # ICEC with vibrationally resolved PI cross sections of D
FC              = 1         # Franck-Condon model to ICEC
bb              = 1         # ICEC for bound-bound transitions of D, starts calculation with resolved or FC
bc              = 0         # ICEC including dissociation of D+, only starts calculation with FC
rydberg         = 0         # Rydberg ICEC for proton-like A, only starts calculation with resolved
calculate       = 1         # activates calculation of ICEC cross sections
plotting        = 1         # generates plots of results from calculation, needs FC calculation
spectra         = 1         # activates the ICEC electron spectrum
cross_section   = 0         # activates total ICEC cross sections
temp_dependence = 0         # activates temperature dependent plots

# === Parameters for electron acceptor (A) and electron donor (D) ===
class unitA():  # A
    name        = 'H'                                           # name of unit A
    deg_factor  = 2 / 1                                         # degeneracy factor g_A/g_A- for detailed balance equation
    IP          = 13.598434599702 * Units.EV2HARTREE            # Ionization potential of A- = electron affinity of A
    file_PI_xs  = os.path.join(DIR_DATA, f'{name}.txt')  # file name of the PI XS of A-
    degree      = 15                                            # degree for polynomial fit, optional
    
class unitD():  # D
    # p.489 in Huber1979
    name        = 'O2'                                   # name of unit D
    IP          = 12.071 * Units.EV2HARTREE              # this is the adiabatic IP and not minimum to minimum, but no time to change the code or the value
    mu          = 7.9974575 * Constants.m_p
    Req         = 1.20752 
    we          = 1580.19 * Units.WAVENUMBER2HARTREE
    wexe        = 11.98 * Units.WAVENUMBER2HARTREE   
    De          = we**2 / 4 / wexe

    file_PI_xs_unresolved = os.path.join(DIR_DATA, f'{name}.txt')     # file name of the unresolved/electronic PI XS of D

class unitDp(): # D+
    name        = 'O2p'
    mu          = 7.9973203 * Constants.m_p
    Req         = 1.1164
    we          = 1904.7 * Units.WAVENUMBER2HARTREE
    wexe        = 16.25 * Units.WAVENUMBER2HARTREE
    De          = we**2 / 4 / wexe