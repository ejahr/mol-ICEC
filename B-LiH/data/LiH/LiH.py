import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from icec.constants import Units, Constants
from calc.fit import generate_linfit

local_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'

# ====================== LiH ==========================

class LiH():
    # Huber p. 382
    mu      = 0.88123833*Constants.m_p
    
    # === https://doi.org/10.1063/1.479970 ===
    # IP_adiabatic = IP_min + Ep_0 - E_0 = 7.68 eV
    IP_min  = 7.743 * Units.EV2HARTREE
    IP      = 7.68 * Units.EV2HARTREE
    
    De      = 2.4924 * Units.EV2HARTREE
    Req     = 3.0148
    alpha   = 0.2124
    we      = 1406.18 * Units.WAVENUMBER2HARTREE
    wexe    = 23.5777 * Units.WAVENUMBER2HARTREE
    v_max   = 2
    morse_parameters = (mu, we, Req, De)

    # --- Photoionization cross section ---
    file_PI_xs_resolved = local_DIR + 'LiH_vi_vf_'
    file_PI_xs_unresolved = local_DIR + 'LiH_PI-xs.txt'
    
    PI_xs = generate_linfit(file_PI_xs_unresolved)
    
    # --- FC Factors for LiH -> LiH+ ---
    FC_abinitio = [[0.0153, 0.0292, 0.0305, 0.0214, 0.0103, 0.0031, 0.0004],
                   [0.0610, 0.0823, 0.0643, 0.0366, 0.0157, 0.0045, 0.0006],
                   [0.1252, 0.1033, 0.0497, 0.0188, 0.0062, 0.0016, 0.0002],
                   [0.1749, 0.0665, 0.0097, 0.0003, 0.0000, 0.0000, 0.0000],
                   [0.1866, 0.0146, 0.0036, 0.0105, 0.0068, 0.0022, 0.0003],
                   [0.1619, 0.0021, 0.0349, 0.0276, 0.0121, 0.0035, 0.0005]]
    
# ====================== LiH+ ==========================

class LiHp():
    mu      = LiH.mu
    # https://doi.org/10.1063/1.479970
    Req     = 4.136
    we      = 442.9 * Units.WAVENUMBER2HARTREE
    alpha   = 0.507 
    wexe    = 42.3 * Units.WAVENUMBER2HARTREE
    De      = we**2 / 4 / wexe

    morse_parameters = (mu, we, Req, De)
    
    v_max = 6