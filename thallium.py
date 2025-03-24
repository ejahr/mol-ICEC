import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/elena/icec-project')
from crosssection.icec.constants import *

DIR = '/home/elena/icec-project/dimers/'
IP_Tl = 6.1 * EV2HARTREE

def angstrom2hartree(wavelength):
    # E = 2 * pi * hbar * c / lambda
    wavelength *= ANGSTROM2BOHR
    energy = 2 * np.pi * c / wavelength
    return energy

def generate_linfit(fname):
    xs_data = np.loadtxt(fname, comments='#')
    energies = xs_data[:,0] 
    energies = angstrom2hartree(energies)
    xs = xs_data[:,1] * MB2AU
    interp_func = sp.interpolate.interp1d(energies, xs, kind='linear', fill_value="extrapolate")
    return interp_func

def PR_xs_Tl(electronE):
    fname = DIR + 'thallium_PI_xs.txt'
    PI_xs_Tl = generate_linfit(fname)
    hbarOmega = electronE + IP_Tl
    deg_factor = 6 / 1 # g_A / g_A+
    PR_xs = deg_factor * hbarOmega**2 / (2*electronE*c**2) * PI_xs_Tl(hbarOmega)
    return PR_xs

e_max = 2.4
energies = np.arange(0.01, e_max, e_max/100) * EV2HARTREE

fig = plt.figure()
ax = plt.gca()
ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon$ [eV]')
ax.set_ylabel(r'$\sigma$ [Mb]')
ax.grid(True)
ax.plot(energies*HARTREE2EV, PR_xs_Tl(energies)*AU2MB, label= r'Thallium PR')

def energy_hbaromega_eV(electronE): 
    return electronE + IP_Tl*HARTREE2EV
def hbaromega_energy_eV(hbaromega):
    return  hbaromega - IP_Tl*HARTREE2EV
secax = ax.secondary_xaxis('top', functions=(energy_hbaromega_eV, hbaromega_energy_eV))
secax.set_xlabel(r'$\epsilon_\text{out}$ [eV]')

ax.legend(loc='upper right')
plt.tight_layout()
fname = DIR + 'plots/Tl_PR.pdf'
fig.savefig(fname)