import numpy as np
import matplotlib.pyplot as plt
from ..icec.constants import Units, Constants
from ..config import DIR
from .fit import generate_polyfit, generate_linfit

# K, Na 2S1/2
degeneracy_2S = 2
# K+, Na+ 1S
degeneracy_1S = 1

IP_K  = 4.34066373 * Units.EV2HARTREE
IP_NA = 5.13907696 * Units.EV2HARTREE
IP_TL = 6.1 * Units.EV2HARTREE

def PR_xs_K(electronE):
    fname = DIR + 'data/K/K_PI.txt'
    PI_xs = generate_polyfit(fname, 10, energy_unit="A")
    hbarOmega = electronE + IP_K
    deg_factor = degeneracy_2S/degeneracy_1S # g_A / g_A+
    PR_xs = deg_factor * hbarOmega**2 / (2*electronE*Constants.c**2) * PI_xs(hbarOmega)
    return PR_xs

def PR_xs_Na(electronE):
    fname = DIR + 'data/Na/Na_PI.txt'
    PI_xs = generate_polyfit(fname, 10, energy_unit="A")
    hbarOmega = electronE + IP_NA
    deg_factor = degeneracy_2S/degeneracy_1S # g_A / g_A+
    PR_xs = deg_factor * hbarOmega**2 / (2*electronE*Constants.c**2) * PI_xs(hbarOmega)
    return PR_xs

def PR_xs_Tl(electronE):
    fname = DIR + 'data/Tl/Tl_PI.txt'
    PI_xs = generate_linfit(fname, energy_unit='A')
    hbarOmega = electronE + IP_TL
    deg_factor = 6 / 1 # g_A / g_A+
    PR_xs = deg_factor * hbarOmega**2 / (2*electronE*Constants.c**2) * PI_xs(hbarOmega)
    return PR_xs

def plot_Tl(energies):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    ax.plot(energies*Units.HARTREE2EV, PR_xs_Tl(energies)*Units.AU2MB, label= r'Thallium PR')

    def energy_hbaromega_eV(electronE): 
        return electronE + IP_TL*Units.HARTREE2EV
    def hbaromega_energy_eV(hbaromega):
        return  hbaromega - IP_TL*Units.HARTREE2EV
    secax = ax.secondary_xaxis('top', functions=(energy_hbaromega_eV, hbaromega_energy_eV))
    secax.set_xlabel(r'$\epsilon_\text{out}$ [eV]')

    ax.legend(loc='upper right')
    plt.tight_layout()
    fname = DIR + 'plots/Tl_PR.pdf'
    fig.savefig(fname)
    
def plot_PR_TlNaK(energies):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.set_title('Photorecombination')
    ax.grid(True)
    ax.plot(energies*Units.HARTREE2EV, PR_xs_Tl(energies)*Units.AU2MB, label= r'Tl+')
    ax.plot(energies*Units.HARTREE2EV, PR_xs_Na(energies)*Units.AU2MB, label= r'K+')
    ax.plot(energies*Units.HARTREE2EV, PR_xs_K(energies)*Units.AU2MB, label= r'Na+')

    ax.legend(loc='upper right')
    plt.tight_layout()
    fname = DIR + 'plots/PR_TlNaK.pdf'
    fig.savefig(fname)

e_max = 2.4
energies = np.arange(0.01, e_max, e_max/200) * Units.EV2HARTREE

plot_Tl(energies)
plot_PR_TlNaK(energies)