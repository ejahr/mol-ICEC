import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import plot.config
from config import DIR
from icec.icec import ICEC
from icec.constants import Units, Constants
from input.fit import generate_polyfit

class ReadOnly(type):
    def __setattr__(self, name, value):
        raise AttributeError("Constants are read-only")
    
# =================== Li ==========================
class Li(metaclass=ReadOnly):
    m = 7*Constants.m_p + 3
    # NIST
    m = 7.0160034366 * Constants.m_p
    r_vdw = 5.2896 # a.u.

# =================== H ==========================
class H(metaclass=ReadOnly):
    # H 2S_1/2
    deg_2S = 2
    deg_factor = deg_2S / 1

    # NIST
    IP = 13.598434599702 * Units.EV2HARTREE
    
    r_vdw = 3.1647 # a.u.
    m = Constants.m_p + 1
    # NIST
    m = 1.00782503223 * Constants.m_p
    
    # Photoionization cross section
    fname = DIR + 'data/H/H.txt'
    PI_xs = generate_polyfit(fname, 15)

    xs_data = np.loadtxt(fname, comments='#')
    E_photon = xs_data[:,0]
    xs = xs_data[:,1]
    coefficients = np.polyfit(E_photon, xs, 15)
    PI_xs_eVMb = np.poly1d(coefficients)
    
    def plot_H_PI_PR(icec:ICEC):
        plot.config.set_rcParams()
        fig = plt.figure()
        ax = plt.gca() 
        ax.set_title('Hydrogen')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\epsilon$ [eV]')
        ax.set_ylabel(r'$\sigma$ [Mb]')
        
        PI_xs = np.array([])
        hbaromega = np.array([])
        for electronE in icec.energyGrid:
            omega = electronE + icec.IP_A
            hbaromega = np.append(hbaromega, [omega*Units.HARTREE2EV])
            xs = icec.PI_xs_A(omega*Units.HARTREE2EV)
            PI_xs = np.append(PI_xs, [xs])
        ax.plot(icec.energyGrid*Units.HARTREE2EV, PI_xs, label = r'$H\to H^+$')
        icec.plot_PR_xs(ax, label = r'$H^+\to H$')
        ax.legend()
        fname = DIR + 'plots/H.PI.PR.pdf'
        fig.savefig(fname)
    
# ====================== LiH ==========================

class LiH(metaclass=ReadOnly):
    # --- Huber ---
    IP = 7.7 * Units.EV2HARTREE # adiabatic?
       
    IP_vert_approx = np.abs(-8.066308039 + 7.770884366)*Units.HARTREE2EV
    IP_min_approx = np.abs(-8.066308039 + 7.78173407)*Units.HARTREE2EV

    m       = H.m + Li.m
    mu      = H.m * Li.m / (H.m + Li.m)
    # Huber p. 382
    mu      = 0.88123833*Constants.m_p
    
    # --- LiH data from https://doi.org/10.1063/1.479970 ---
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
    #print(mu, Req, De, we)

    # Table III
    energy_v0 = 697.72 * Units.WAVENUMBER2HARTREE
    vib_spacing  = np.array([0, 1359.66, 1314.68, 1270.55, 1227.31, 1184.87, 1143.06, 1101.72, 1060.73, 1019.88, 978.85, 937.40, 895.21, 851.75, 806.39, 758.32, 706.47, 649.46, 585.50, 512.30, 427.12, 326.95, 209.30, 76.29])
    vib_spacing *= Units.WAVENUMBER2HARTREE
    vib_diff_to_v0 = np.cumsum(vib_spacing)
    vib_energies = vib_diff_to_v0 + energy_v0
    
    # --- Photoionization cross section ---
    file_PI_xs_resolved = DIR + 'data/LiH/LiH_vi_vf_'
    file_PI_xs_unresolved = DIR + 'data/LiH/LiH'

    xs_data = np.loadtxt(file_PI_xs_unresolved + '.txt', comments='#')
    energies = xs_data[:,0]
    xs = xs_data[:,1]
    PI_xs_eVMb = sp.interpolate.interp1d(energies, xs, kind='linear', fill_value="extrapolate")

    max_kinE_unresolved = energies[-1]*Units.EV2HARTREE - H.IP

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

# ===================== H+ = LiH =================
class Hp_LiH(metaclass=ReadOnly):
    R_min_vdw = (Li.r_vdw + H.r_vdw + LiH.Req)/2 + H.r_vdw

    input_electronic = (H.deg_factor, H.IP*Units.HARTREE2EV, LiH.IP*Units.HARTREE2EV, H.PI_xs_eVMb, LiH.PI_xs_eVMb)

    input = [H.deg_factor, H.IP, LiH.IP, H.PI_xs, LiH.file_PI_xs_resolved]
    input_unresolved = [H.deg_factor, H.IP, LiH.IP, H.PI_xs, LiH.file_PI_xs_unresolved]
    
    def R_min():
        # https://doi.org/10.1039/D3CP02959J Tab.3
        r_LiH = 1.646 * Units.ANGSTROM2BOHR
        r_HH = 2.513 * Units.ANGSTROM2BOHR
        r_COM_LiH = (H.m * r_LiH + 0) / (H.m + Li.m)
        R_min = r_LiH - r_COM_LiH + r_HH
        return R_min

# ==================== Test =====================
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

    # min R between H and LiH
    print(f"Hp_LiH.R_min = {round(Hp_LiH.R_min_vdw,5)} a.u. = {round(Hp_LiH.R_min_vdw*Units.BOHR2ANGSTROM,5)} A")
    print(f"LiH.r_mu     = {round(LiH.r_mu,5)} a.u. = {round(LiH.r_mu*Units.BOHR2ANGSTROM,5)} A")
    R_min_COM = (H.r_vdw + Li.r_vdw + LiH.r_mu)*Units.BOHR2ANGSTROM 
    print(f"R_COM H-LiH  = {round(R_min_COM,5)} A")
    R_min_COM = (H.r_vdw + H.r_vdw + (LiH.Req - LiH.r_mu))*Units.BOHR2ANGSTROM 
    print(f"R_COM H-HLi  = {round(R_min_COM,5)} A")
    
    r_mu = (H.m * 1.646*Units.ANGSTROM2BOHR + 0) / (H.m + Li.m)
    print(f"\nr_COM LiH    = {round(r_mu,5)} a.u. = {round(r_mu*Units.BOHR2ANGSTROM,5)} A")
    print(f"R_COM LiH-H+ = {round(Hp_LiH.R_min(),5)} a.u. = {round(Hp_LiH.R_min()*Units.BOHR2ANGSTROM,5)} A")
    
    #r_mu = (H.m * 1.6*Units.ANGSTROM2BOHR + 0) / (H.m + Li.m)
    #print(f"r_COM LiH    = {round(r_mu,5)} a.u. = {round(r_mu*Units.BOHR2ANGSTROM,5)} A")
    #R_min_COM = 1.6 - r_mu*Units.BOHR2ANGSTROM + 2.6
    #print(f"R_min LiH-H+ = {round(R_min_COM,5)} A")
    
    #R_min_COM   = LiH.r_mu*Units.BOHR2ANGSTROM + 5
    #print(f"R_min HLi-H+ = {round(R_min_COM,5)} A")
    
    #r_mu        = (Constants.m_p * 2.1*Units.ANGSTROM2BOHR + 0) / (Constants.m_p + Li.m)
    #print(f"r_COM LiH+   = {round(r_mu,5)} a.u. = {round(r_mu*Units.BOHR2ANGSTROM,5)} A")
    #R_min_COM   = 2.1 - r_mu*Units.BOHR2ANGSTROM + 3
    #print(f"R_min LiH+-H = {round(R_min_COM,5)} A")
    
