import os
import numpy as np
from icec.constants import Units, Constants

'''
All parameters should be given (or converted to) atomic units.
'''

DIR = os.path.dirname(os.path.realpath(__file__)) + '/' # DIR where this config.py file is located
DIR_DATA = DIR + 'HLiH/data/'                           # dir name where PI XS data is located
DIR_RESULTS = DIR + 'HLiH/results/'                     # dir name where results should be saved
DIR_PLOTS = DIR + 'HLiH/plots/'                         # dir name where plots should be saved       

# === System ===
system_name = 'Hp-LiH'                                  # used in the header of result files
title       = r'$\text{H}^+ \text{LiH}$'                # plot title
reaction    = 'e- + H+ + LiH -> H + LiH+ + e-'          # used in the header of result files

# === Parameters for the calculations ===
R           = 3.9522557993362915 * Units.ANGSTROM2BOHR   # distance between center of masses of A and D
L           = 8     * Units.ANGSTROM2BOHR   # box length for discretizing the dissociative states of D+

vD_max_bc   = 7                             # maximum initial vibrational state considered for bound-continuum transitions
min_kinE    = 0.01  * Units.EV2HARTREE      # minimum incoming electron energy
max_kinE    = 9     * Units.EV2HARTREE      # maximum incoming electron energy
max_dissE   = 2     * Units.EV2HARTREE      # maximum energy up to which the dissociative states are calculated
num_grid    = 1000                          # number of grid points between min and max electron energies
n_max       = 10                            # number of rydberg states

electronE   = 1     * Units.EV2HARTREE      # incoming electron energy for ICEC electron spectra
T           = [15, 300, 1500]   # Kelvin    # temperatures for temperature dependent spectra

# === Define which calculations are active ===
calc_roots      = 0         # activates calculation of the dissociative states of D+ within box of L, needs to run at least once
resolved        = 1         # ICEC with vibrationally resolved PI cross sections of D
FC              = 1         # Franck-Condon model to ICEC
bb              = 1         # ICEC for bound-bound transitions of D, starts calculation with resolved or FC
bc              = 0         # ICEC including dissociation of D+, only starts calculation with FC
rydberg         = 0         # Rydberg ICEC, only starts calculation with resolved
calculate       = 1         # activates calculation of ICEC cross sections
plotting        = 0         # generates plots of results from calculation, needs FC calculation
spectra         = 0         # activates the ICEC electron spectrum
cross_section   = 0         # activates total ICEC cross sections
temp_dependence = 0         # activates temperature dependent plots

# === Parameters for electron acceptor (A) and electron donor (D) ===
class unitA():  # A
    name        = 'H'                                   # name of unit A
    deg_factor  = 2 / 1                                 # degeneracy factor g_A/g_A- for detailed balance equation
    IP          = 13.598434599702 * Units.EV2HARTREE    # Ionization potential of A- = electron affinity of A
    file_PI_xs  = DIR_DATA + f'{name}/{name}.txt'       # file name of the PI XS of A-
    degree      = 15                                    # degree for polynomial fit, optional
    
class unitD():  # D
    name        = 'LiH'                                 # name of unit D
    IP          = 7.743 * Units.EV2HARTREE              # energy difference between minima of initial and final PES
    mu          = 0.88123833*Constants.m_p
    De          = 2.4924 * Units.EV2HARTREE
    Req         = 3.0148 
    we          = 1406.18 * Units.WAVENUMBER2HARTREE
    wexe        = 23.5777 * Units.WAVENUMBER2HARTREE    # optional if De is given
    v_max       = 2                                     # maximum vibrational state for which the resolved PI XS is available

    # energy spacings between initial vibrational states, optional
    vib_spacing     = np.array([0, 1359.66, 1314.68, 1270.55, 1227.31, 1184.87, 1143.06, 
                             1101.72, 1060.73, 1019.88, 978.85, 937.40, 895.21, 851.75, 
                             806.39, 758.32, 706.47, 649.46, 585.50, 512.30, 427.12, 
                             326.95, 209.30, 76.29]) * Units.WAVENUMBER2HARTREE
    
    file_PI_xs_resolved = DIR_DATA + f'{name}/{name}_vi_vf_'    # file name structure of the resolved PI XS of D, in code: + '_0_0.txt'
    file_PI_xs_unresolved = DIR_DATA + f'{name}/{name}.txt'     # file name of the unresolved/electronic PI XS of D

class unitDp(): # D+
    mu          = unitD.mu
    Req         = 4.136
    we          = 442.9 * Units.WAVENUMBER2HARTREE
    wexe        = 42.3 * Units.WAVENUMBER2HARTREE
    De          = we**2 / 4 / wexe
    v_max       = 6                                             # maximum vibrational state for which the resolved PI XS is available
    
    # energy spacings between final vibrational states, optional
    vib_spacing = np.array([0, 351.6, 257.2, 163.5, 84.1, 31.8, 7.3])*Units.WAVENUMBER2HARTREE 



