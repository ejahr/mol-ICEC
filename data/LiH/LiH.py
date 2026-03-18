import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import DIR
from icec.constants import Units, Constants
from data.fit import generate_linfit
from data.H.H import H

class ReadOnly(type):
    def __setattr__(self, name, value):
        raise AttributeError("Constants are read-only")
    
# =================== Li ==========================
class Li(metaclass=ReadOnly):
    m = 7*Constants.m_p + 3
    # NIST
    m = 7.0160034366 * Constants.m_p
    r_vdw = 5.2896 # a.u.

# ====================== LiH ==========================

class LiH(metaclass=ReadOnly):
    # Huber p. 382
    IP = 7.7 * Units.EV2HARTREE # adiabatic?

    m       = H.m + Li.m
    mu      = H.m * Li.m / (H.m + Li.m)
    # Huber p. 382
    mu      = 0.88123833*Constants.m_p
    
    # --- https://doi.org/10.1063/1.479970 ---
    energy_diff_at_inf = np.abs(7.974721285 - 7.776735464) # energy difference at R=inf
    IP_vert_approx = np.abs(-8.066308039 + 7.770884366)
    IP_min_approx = np.abs(-8.066308039 + 7.78173407)
    IP      = 7.743 * Units.EV2HARTREE
    # IP + Ep_0 - E_0 = 7.68 eV
    
    # Table IV
    E_min   = -8.021321 
    De      = 2.4924 * Units.EV2HARTREE
    Req     = 3.0148
    alpha   = 0.2124
    we      = 1406.18 * Units.WAVENUMBER2HARTREE
    wexe    = 23.5777 * Units.WAVENUMBER2HARTREE
    v_max   = 2
    morse_parameters = (mu, we, Req, De)

    r_mu    = (H.m*Req + Li.m*0) / (H.m + Li.m)

    #print('alpha =', alpha)
    #print('we * sqrt(mu/2/De) =', we * np.sqrt(mu/2/De))

    # Table III
    energy_v0 = 697.72 * Units.WAVENUMBER2HARTREE
    vib_spacing  = np.array([0, 1359.66, 1314.68, 1270.55, 1227.31, 1184.87, 1143.06, 
                             1101.72, 1060.73, 1019.88, 978.85, 937.40, 895.21, 851.75, 
                             806.39, 758.32, 706.47, 649.46, 585.50, 512.30, 427.12, 
                             326.95, 209.30, 76.29])
    vib_spacing *= Units.WAVENUMBER2HARTREE
    vib_diff_to_v0 = np.cumsum(vib_spacing)
    vib_energies = vib_diff_to_v0 + energy_v0
    
    # --- Photoionization cross section ---
    file_PI_xs_resolved = DIR + 'data/LiH/LiH_vi_vf_'
    file_PI_xs_unresolved = DIR + 'data/LiH/LiH.txt'
    
    PI_xs = generate_linfit(file_PI_xs_unresolved)

    xs_data = np.loadtxt(file_PI_xs_unresolved, comments='#')
    energies = xs_data[:,0]
    max_kinE_unresolved = energies[-1]*Units.EV2HARTREE - H.IP
    
    # --- FC Factors for LiH -> LiH+ ---
    # https://doi.org/10.1063/1.479970
    FC_abinitio = [[0.0153, 0.0292, 0.0305, 0.0214, 0.0103, 0.0031, 0.0004],
                   [0.0610, 0.0823, 0.0643, 0.0366, 0.0157, 0.0045, 0.0006],
                   [0.1252, 0.1033, 0.0497, 0.0188, 0.0062, 0.0016, 0.0002],
                   [0.1749, 0.0665, 0.0097, 0.0003, 0.0000, 0.0000, 0.0000],
                   [0.1866, 0.0146, 0.0036, 0.0105, 0.0068, 0.0022, 0.0003],
                   [0.1619, 0.0021, 0.0349, 0.0276, 0.0121, 0.0035, 0.0005]]
    
# ====================== LiH+ ==========================

class LiHp(metaclass=ReadOnly):
    m_Lip   = 7*Constants.m_p + 2
    mu      = H.m * m_Lip / (H.m + m_Lip)
    mu      = LiH.mu
    # https://doi.org/10.1063/1.479970
    Req     = 4.136
    we      = 442.9 * Units.WAVENUMBER2HARTREE
    alpha   = 0.507 
    wexe    = 42.3 * Units.WAVENUMBER2HARTREE
    De      = we**2 / 4 / wexe

    morse_parameters = (mu, we, Req, De)
    
    v_max = 6

    # Table V
    vib_spacing = np.array([0, 351.6, 257.2, 163.5, 84.1, 31.8, 7.3]) 
    vib_spacing *= Units.WAVENUMBER2HARTREE
    vib_diff_to_v0 = np.cumsum(vib_spacing)
    
# ====================== TEST ==========================

if __name__ == "__main__":
    #print('Vertical Ionization potential:', LiH.IP_vert)
    print(f'E_p(Re)  -E(Re) = {LiH.IP_vert_approx} a.u.')
    print(f'E_p(Re_p)-E(Re) = {LiH.IP_min_approx} a.u.\n')

    # Huber p. 382
    print("De Lundsgaard", LiH.De*Units.HARTREE2EV, "eV")
    print("m  calculated", H.m * Li.m / (H.m + Li.m))
    print("m  Huber     ", LiH.mu)
    print("w  Lundsgaard", LiH.we*Units.HARTREE2EV, "eV")
    print("w  Huber     ", 1405.65*Units.WAVENUMBER2HARTREE*Units.HARTREE2EV, "eV")
    print("Re Lundsgaard", LiH.Req)
    print("Re Huber     ", 1.5957*Units.ANGSTROM2BOHR, "\n")