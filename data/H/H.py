import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import plot.config
from config import DIR
from icec.icec import ICEC
from icec.constants import Units, Constants
from calc.fit import generate_polyfit

class ReadOnly(type):
    def __setattr__(self, name, value):
        raise AttributeError("Constants are read-only")
    
class H(metaclass=ReadOnly):
    # H 2S_1/2
    deg_2S = 2
    deg_factor = deg_2S / 1

    # NIST
    IP = 13.598434599702 * Units.EV2HARTREE
    # Rydberg R_H \approx IP
    
    r_vdw = 3.1647 # a.u.
    m = Constants.m_p + 1
    # NIST
    m = 1.00782503223 * Constants.m_p
    
    # Photoionization cross section
    fname = DIR + 'data/H/H.txt'
    PI_xs = generate_polyfit(fname, 15)

    def plot_H_PI_PR(icec:ICEC):
        plot.config.set_rcParams()
        fig = plt.figure()
        ax = plt.gca() 
        ax.set_title('Hydrogen')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\epsilon$ [eV]')
        ax.set_ylabel(r'$\sigma$ [Mb]')
        
        energies = icec.energyGrid
        hbarOmega = np.array([icec.hbarOmega(electronE) for electronE in energies])
        PI_xs = np.array([icec.PI_xs_A(omega) for omega in hbarOmega])

        ax.plot(energies*Units.HARTREE2EV, PI_xs*Units.AU2MB, label = r'$H\to H^+$')
        icec.plot_PR_xs(ax, label = r'$H^+\to H$')
        ax.legend()
        fname = DIR + 'plots/H.PI.PR.pdf'
        fig.savefig(fname)