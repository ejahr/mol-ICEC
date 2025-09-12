import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ..icec.constants import *

DIR = '/home/elena/intraICEC/dimers/'

def generate_polyfit(fname, degree):
    xs_data = np.loadtxt(fname, comments='#')
    E_photon = xs_data[:,0] * EV2HARTREE
    xs = xs_data[:,1] * MB2AU
    coefficients = np.polyfit(E_photon, xs, degree)
    #test_polyfit(np.poly1d(coefficients), E_photon, xs, degree)
    return np.poly1d(coefficients)

def test_polyfit(polyfit, x, y, degree):
    x_fit = np.linspace(min(x), 20*EV2HARTREE, 100)
    y_fit = polyfit(x_fit)
    plt.xlim(min(x)*HARTREE2EV, 20)
    plt.scatter(x*HARTREE2EV, y*AU2MB, color='red', label='Data points')  # Original data points
    plt.plot(x_fit*HARTREE2EV, y_fit*AU2MB, label=f'Polynomial fit, degree = ' + str(degree))  # Fitted curve
    plt.legend()
    fname = DIR + 'H/test_polyfit.pdf'
    plt.savefig(fname)
    
def generate_linfit(fname):
    xs_data = np.loadtxt(fname, comments='#')
    energies = xs_data[:,0] * EV2HARTREE
    xs = xs_data[:,1] * MB2AU
    interp_func = sp.interpolate.interp1d(energies, xs, kind='linear', fill_value="extrapolate")
    #test_linfit(interp_func, energies, xs)
    return interp_func

def test_linfit(interp_func, x, y):
    x_fit = np.geomspace(min(x), 20*EV2HARTREE, 1000)
    y_fit = interp_func(x_fit)
    plt.xlim(min(x)*HARTREE2EV, 20)
    plt.scatter(x*HARTREE2EV, y*AU2MB, color='red', label='Data points') 
    plt.plot(x_fit*HARTREE2EV, y_fit*AU2MB, label=f'Linear fit')  
    plt.legend()
    fname = DIR + 'B/test_linfit.pdf'
    plt.savefig(fname)

# =================== H+ ==========================
# H 2S1/2
deg_2S = 2

IP_H = 13.598434599702 * EV2HARTREE
deg_factor_H = deg_2S / 1

fname = DIR + 'H/H.txt'
PI_xs_H = generate_polyfit(fname, 15)

# =================== B+ ==========================
deg_2P = 6
deg_1S = 1
deg_factor_B = 6
IP_B = 8.298019 * EV2HARTREE

fname = DIR + 'B/B.txt'
PI_xs_B = generate_linfit(fname)

# ====================== LiH ==========================
IP_LiH = 7.7 * EV2HARTREE
IP_LiH = 7.9 * EV2HARTREE

m_p =  1836.152673426

#  https://doi.org/10.1063/1.479970
m_H     = m_p + 1
m_Li    = 7*m_p + 3

mu      = m_H * m_Li / (m_H + m_Li)
E_min   = -8.021321 
De      = 2.4924 * EV2HARTREE
Req     = 3.0148
alpha   = 0.2124 * WAVENUMBER2HARTREE
we      = 1406.18 * WAVENUMBER2HARTREE
wexe    = 23.5777 * WAVENUMBER2HARTREE

print(alpha)
print(we * np.sqrt(mu/2/De))

stateX = (mu, we, Req, De)

energy_v0 = 697.72 * WAVENUMBER2HARTREE
vib_spacing_LiH  = np.array([0, 1359.66, 1314.68, 1270.55, 1227.31, 1184.87, 1143.06, 1101.72, 1060.73, 1019.88, 978.85, 937.40, 895.21, 851.75, 806.39, 758.32, 706.47, 649.46, 585.50, 512.30, 427.12, 326.95, 209.30, 76.29]) * WAVENUMBER2HARTREE
vib_energies_LiH = vib_spacing_LiH + energy_v0
vib_diff_to_ground_LiH = np.cumsum(vib_spacing_LiH)

#https://doi.org/10.1063/1.479970
mu      = m_H * m_Li / (m_H + m_Li)
Req     = 4.136
we      = 442.9 * WAVENUMBER2HARTREE
alpha   = 0.507 * WAVENUMBER2HARTREE
wexe    = 42.3 * WAVENUMBER2HARTREE

vib_spacing_LiHp = np.array([0, 351.6, 257.2, 163.5, 84.1, 31.8, 7.3]) * WAVENUMBER2HARTREE
vib_diff_to_ground_LiHp = np.cumsum(vib_spacing_LiHp)

file_PI_xs_LiH = DIR + 'LiH/LiH_vi_vf_'

# ===================== H+ = LiH =================

input_HLiH = [deg_factor_H, IP_H, IP_LiH, PI_xs_H, file_PI_xs_LiH]

# ===================== H+ = LiH =================

input_BLiH = [deg_factor_B, IP_B, IP_LiH, PI_xs_B, file_PI_xs_LiH]