import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from icec.constants import Units, Constants

DIR = '/home/elena/intraICEC/dimers/'

def generate_polyfit(fname, degree):
    xs_data = np.loadtxt(fname, comments='#')
    E_photon = xs_data[:,0] * Units.EV2HARTREE
    xs = xs_data[:,1] * Units.MB2AU
    coefficients = np.polyfit(E_photon, xs, degree)
    #test_polyfit(np.poly1d(coefficients), E_photon, xs, degree)
    return np.poly1d(coefficients)

def test_polyfit(polyfit, x, y, degree):
    x_fit = np.linspace(min(x), 20*Units.EV2HARTREE, 100)
    y_fit = polyfit(x_fit)
    plt.xlim(min(x)*Units.HARTREE2EV, 20)
    plt.scatter(x*Units.HARTREE2EV, y*Units.AU2MB, color='red', label='Data points')  # Original data points
    plt.plot(x_fit*Units.HARTREE2EV, y_fit*Units.AU2MB, label=f'Polynomial fit, degree = ' + str(degree))  # Fitted curve
    plt.legend()
    fname = DIR + 'data/H/test_polyfit.pdf'
    plt.savefig(fname)
    
def generate_linfit(fname):
    xs_data = np.loadtxt(fname, comments='#')
    energies = xs_data[:,0] * Units.EV2HARTREE
    xs = xs_data[:,1] * Units.MB2AU
    interp_func = sp.interpolate.interp1d(energies, xs, kind='linear', fill_value="extrapolate")
    #test_linfit(interp_func, energies, xs)
    return interp_func

def test_linfit(interp_func, x, y):
    x_fit = np.geomspace(min(x), 20*Units.EV2HARTREE, 1000)
    y_fit = interp_func(x_fit)
    plt.xlim(min(x)*Units.HARTREE2EV, 20)
    plt.scatter(x*Units.HARTREE2EV, y*Units.AU2MB, color='red', label='Data points') 
    plt.plot(x_fit*Units.HARTREE2EV, y_fit*Units.AU2MB, label=f'Linear fit')  
    plt.legend()
    fname = DIR + 'data/B/test_linfit.pdf'
    plt.savefig(fname)

class ReadOnly(type):
    def __setattr__(self, name, value):
        raise AttributeError("Constants are read-only")

# =================== H+ ==========================
class H(metaclass=ReadOnly):
    # H 2S1/2
    deg_2S = 2
    deg_factor = deg_2S / 1

    # NIST
    IP = 13.598434599702 * Units.EV2HARTREE
    
    r_vdw = 3.1647
    m = Constants.m_p + 1
    
    # Photoionization cross section
    fname = DIR + 'data/H/H.txt'
    PI_xs = generate_polyfit(fname, 15)

    xs_data = np.loadtxt(fname, comments='#')
    E_photon = xs_data[:,0]
    xs = xs_data[:,1]
    coefficients = np.polyfit(E_photon, xs, 15)
    PI_xs_eVMb = np.poly1d(coefficients)

# =================== B+ ==========================
class B:
    deg_2P = 6
    deg_1S = 1
    deg_factor = 6
    IP = 8.298019 * Units.EV2HARTREE

    fname = DIR + 'data/B/B.txt'
    PI_xs = generate_linfit(fname)

# ====================== LiH ==========================

class LiH(metaclass=ReadOnly):
    # Huber
    IP = 7.7 * Units.EV2HARTREE # adiabatic?
       
    IP_vert_approx = np.abs(-8.066308039 + 7.770884366)*Units.HARTREE2EV
    IP_min_approx = np.abs(-8.066308039 + 7.78173407)*Units.HARTREE2EV

    m_Li    = 7*Constants.m_p + 3
    mu      = H.m * m_Li / (H.m + m_Li)
    # Huber p. 382
    mu = 0.88123833*Constants.m_p
    # https://doi.org/10.1063/1.479970
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

    r_mu    = (H.m*Req + m_Li*0) / (H.m + m_Li)

    #print('alpha =', alpha)
    #print('we * sqrt(mu/2/De) =', we * np.sqrt(mu/2/De))
    #print(mu, Req, De, we)

    # Table III
    energy_v0 = 697.72 * Units.WAVENUMBER2HARTREE
    vib_spacing  = np.array([0, 1359.66, 1314.68, 1270.55, 1227.31, 1184.87, 1143.06, 1101.72, 1060.73, 1019.88, 978.85, 937.40, 895.21, 851.75, 806.39, 758.32, 706.47, 649.46, 585.50, 512.30, 427.12, 326.95, 209.30, 76.29])
    vib_spacing *= Units.WAVENUMBER2HARTREE
    vib_diff_to_v0 = np.cumsum(vib_spacing)
    vib_energies = vib_diff_to_v0 + energy_v0
    
    # Photoionization cross section
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
    #print(De*Units.HARTREE2EV)
    
    v_max = 6

    # Table V
    vib_spacing = np.array([0, 351.6, 257.2, 163.5, 84.1, 31.8, 7.3]) 
    vib_spacing *= Units.WAVENUMBER2HARTREE
    vib_diff_to_v0 = np.cumsum(vib_spacing)

# ===================== H+ = LiH =================
class Hp_LiH():
    r_vdw_Li = 5.2896
    R_min = (r_vdw_Li + H.r_vdw + LiH.Req)/2 + H.r_vdw
    #print("R_min", R_min, R_min*Units.BOHR2ANGSTROM)

    input_electronic = (H.deg_factor, H.IP*Units.HARTREE2EV, LiH.IP*Units.HARTREE2EV, H.PI_xs_eVMb, LiH.PI_xs_eVMb)

    input = [H.deg_factor, H.IP, LiH.IP, H.PI_xs, LiH.file_PI_xs_resolved]
    input_unresolved = [H.deg_factor, H.IP, LiH.IP, H.PI_xs, LiH.file_PI_xs_unresolved]

# ===================== B+ = LiH =================
class Bp_LiH:
    input = [B.deg_factor, B.IP, LiH.IP, B.PI_xs, LiH.file_PI_xs_resolved]
    
    
#print('Vertical Ionization potential:', LiH.IP_vert)
print('Approx E_p(Re)  -E(Re):', LiH.IP_vert_approx)
print('Approx E_p(Re_p)-E(Re):', LiH.IP_min_approx)

# Huber p. 382
print("De Lundsgaard", LiH.De*Units.HARTREE2EV)
print("m calculated ", H.m * LiH.m_Li / (H.m + LiH.m_Li))
print("m Huber      ", 0.88123833*Constants.m_p)
print("w Lundsgaard ", LiH.we*Units.HARTREE2EV)
print("w Huber      ", 1405.65*Units.WAVENUMBER2HARTREE*Units.HARTREE2EV)
print("Re Lundsgaard", LiH.Req)
print("Re Huber     ", 1.5957*Units.ANGSTROM2BOHR)