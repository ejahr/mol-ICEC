import os
import numpy as np
from icec.constants import Units, Constants
import calc.fit

'''
All parameters should be given (or converted to) atomic units.
'''

# get local DIR to keep file structure consistent
DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
DIR_DATA = DIR + 'HLiH/data/'
DIR_RESULTS = DIR + 'HLiH/results/'
DIR_PLOTS = DIR + 'HLiH/plots/'

# === System ===
system_name = 'Hp-LiH'
title       = r'$\text{H}^+ \text{LiH}$'            # plot title
reaction    = 'e- + H+ + LiH -> H + LiH+ + e-'

# === Parameters for the calculations ===
R           = 3.95  * Units.ANGSTROM2BOHR
L           = 8     * Units.ANGSTROM2BOHR

vD_max_bc   = 7                             # maximum initial vibrational state considered for bound-continuum transitions
min_kinE    = 0.01  * Units.EV2HARTREE      
max_kinE    = 9     * Units.EV2HARTREE
max_dissE   = 2     * Units.EV2HARTREE
num_grid    = 1000
n_max       = 10     # number of rydberg states

electronE   = 1     * Units.EV2HARTREE      # incoming electron energy for spectrum
T           = [15, 300, 1500]   # Kelvin    # temperatures for temperature dependent spectra

# === Define which calculations are active ===
calc_roots      = 0
bb              = 1
bc              = 0
FC              = 1
plotting        = 0
calculate       = 1
spectra         = 1
cross_section   = 0
temp_dependence = 0
plot_info       = 1
print_info      = 0