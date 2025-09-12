import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ..icec.constants import *
from ..fitting import generate_polyfit, generate_linfit

DIR = '/home/elena/intraICEC/dimers/'

# K, Na 2S1/2
degeneracy_2S = 2
# K+, Na+ 1S
degeneracy_1S = 1

IP_K  = 4.34066373 * EV2HARTREE
IP_NA = 5.13907696 * EV2HARTREE
IP_TL = 6.1 * EV2HARTREE

def PR_xs_K(electronE):
    fname = DIR + 'data/K/K_PI.txt'
    PI_xs = generate_polyfit(fname, 10, energy_unit="A")
    hbarOmega = electronE + IP_K
    deg_factor = degeneracy_2S/degeneracy_1S # g_A / g_A+
    PR_xs = deg_factor * hbarOmega**2 / (2*electronE*c**2) * PI_xs(hbarOmega)
    return PR_xs

def PR_xs_Na(electronE):
    fname = DIR + 'data/Na/Na_PI.txt'
    PI_xs = generate_polyfit(fname, 10, energy_unit="A")
    hbarOmega = electronE + IP_NA
    deg_factor = degeneracy_2S/degeneracy_1S # g_A / g_A+
    PR_xs = deg_factor * hbarOmega**2 / (2*electronE*c**2) * PI_xs(hbarOmega)
    return PR_xs

def PR_xs_Tl(electronE):
    fname = DIR + 'data/Tl/Tl_PI.txt'
    PI_xs = generate_linfit(fname, energy_unit='A')
    hbarOmega = electronE + IP_TL
    deg_factor = 6 / 1 # g_A / g_A+
    PR_xs = deg_factor * hbarOmega**2 / (2*electronE*c**2) * PI_xs(hbarOmega)
    return PR_xs


e_max = 2.4
energies = np.arange(0.01, e_max, e_max/200) * EV2HARTREE

fig = plt.figure()
ax = plt.gca()
ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon$ [eV]')
ax.set_ylabel(r'$\sigma$ [Mb]')
ax.set_title('Photorecombination')
ax.grid(True)
ax.plot(energies*HARTREE2EV, PR_xs_Tl(energies)*AU2MB, label= r'Tl+')
ax.plot(energies*HARTREE2EV, PR_xs_Na(energies)*AU2MB, label= r'K+')
ax.plot(energies*HARTREE2EV, PR_xs_K(energies)*AU2MB, label= r'Na+')

ax.legend(loc='upper right')
plt.tight_layout()
fname = DIR + 'plots/PR_TlNaK.pdf'
fig.savefig(fname)