''' 
Defines functions for generating cross section plots:
- bb_resolved_FC_rydberg: ICEC cross section vs. incoming electron energy. Only bound-bound transitions of D from vi=0. 
- bb_and_bc: ICEC cross section vs. incoming electron energy. Includes dissociative states of D+.
- bb_vD: ICEC cross section vs. incoming electron energy for different initial vibrational states of D.
- boltzmann_bb_and_bc: ICEC cross section vs. incoming electron energy for different temperatures.
'''

import numpy as np
import matplotlib.pyplot as plt
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.constants import Units, Constants
import calc.file_io as file_io
from config import DIR_PLOTS, unitA
from plot.config import set_rcParams

set_rcParams()
width, height = 6, 4

# ===== HELPER FUNCTIONS =====

def set_axes(ax):
    "log yscale, labels: epsilon, sigma"
    ax.set_yscale('log')
    ax.set_xlabel(r'$\varepsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    
# ===== HELPER PLOT FUNCTIONS =====
            
def plot_xs_el(ax, system, R, label='electronic', color='black', **kwargs):
    'plots electronic ICEC cross section against incoming electron energies'
    results = file_io.read_xs(system, R, modifier='-electronic')
    energy = results[:,0]
    xs = results[:,1]
    ax.plot(energy, xs, color=color, label=label, **kwargs)

def plot_xs_bb(ax, system, R, vD, label='icec', modifier='', **kwargs):
    'plots ICEC cross section against incoming electron energies for vD -> bound states'
    if modifier == '':
      ax.set_xlim(-0.2, 8.6)  
    results = file_io.read_xs(system, R, modifier + '.bb')
    ax.plot(results[:,0], results[:, vD+1], label=label, **kwargs)
    
def plot_xs_rydberg(ax, system, R, n=2, **kwargs):
    results = file_io.read_xs(system, R, '-rydberg.bb')
    energies = results[:,0]
    #n_max = len(results[0,:]) - 1
    tot_results = results[:,n-1]
    #for n in range(2, n_max+1):
    #    tot_results += results[:,n]
    ax.plot(energies, tot_results, label=r"b-b Ryd", **kwargs)
    
def plot_xs_bc(ax, system, R, vD, L, label='icec', modifier='', **kwargs):
    'plots ICEC cross section against incoming electron energies for vD -> dissociative states'
    results = file_io.read_xs(system, R, modifier + '.bc', L)
    ax.plot(results[:,0], results[:, vD+1], label=label, **kwargs)
    
def plot_xs_tot(ax, system, R, vD, L, label='icec', modifier='', **kwargs):
    'plots ICEC cross section against incoming electron energies for all transitions from vD'
    results_bb = file_io.read_xs(system, R, modifier + '.bb')
    results_bc = file_io.read_xs(system, R, modifier + '.bc', L)
    results = results_bb[:, vD+1] + results_bc[:, vD+1]
    ax.plot(results_bb[:,0], results, label=label, **kwargs)
    
# ===== CROSS SECTION PLOTS ======

def PI_PR_A(icec:ICEC):
    ''' 
    Generates plot: 
        photoionization and photorecombination cross section of unit A
    '''
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_yscale('log')
    ax.set_xlabel(r'$\varepsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    
    energies = icec.energyGrid
    omegas = np.array([icec.omega(electronE) for electronE in energies])
    PI_xs = np.array([icec.PI_xs_A(omega) for omega in omegas])

    ax.plot(energies*Units.HARTREE2EV, PI_xs*Units.AU2MB, label = 'PI')
    icec.plot_PR_xs(ax, label = 'PR')
    ax.legend()
    fname = DIR_PLOTS + f'{unitA.name}.PI.PR.pdf'
    fig.savefig(fname)
    
def bb_resolved_FC_rydberg(system, icec:IntraICEC, R, vi:int = 0):
    ''' 
    Generates plot: 
        ICEC cross section vs. incoming electron energy.
        Only bound-bound transitions of D from vi=0 are considered. 
        Results from Rydberg, vibrationally resolved, and Franck-Condon model. Optional electronic results.
    
    Legend
        Black: electronic results with vertical ionization of D
        red (b-b): with vibrationally resolved cross sections of D
        blue (b-b FC): based on Franck-Condon model
        purple (b-b Ryd): A captures into the n=2 Rydberg state.
        dotted grey (PR): photorecombination cross section of A
    '''
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    set_axes(ax)
    if R < 5*Units.ANGSTROM2BOHR:
        ax.set_ylim(1e-4,1e3)
        
    plot_xs_bb(ax, system, R, vi, label=r'b-b', color='tab:red')
    plot_xs_bb(ax, system, R, vi, label=r'b-b FC', modifier='-FC', color='tab:blue', zorder=1)
    plot_xs_rydberg(ax, system, R, n=2, color='tab:purple')
    plot_xs_el(ax, system, R)
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='dimgray', ls=':') 
    
    ax.legend(ncol=2)
    fname = DIR_PLOTS + f'{system}.xs.bb.v{vi}.R{round(R*Units.BOHR2ANGSTROM)}.pdf'
    plt.tight_layout(pad=0.5)
    fig.savefig(fname)
    
def bb_and_bc(system, icec:IntraICEC, R, vi:int = 0):
    ''' 
    Generates plot: 
        ICEC cross section vs. incoming electron energy.
        Bound-bound and bound-dissociative transitions of D from vi=0 are considered. 
        Results from Franck-Condon model. Optional electronic results.
    
    Legend
        black (elec.): electronic results 
        solid blue (b-b): includes bound-bound transitions of D 
        dashed blue (b-d): includes bound-dissociative transitions of D
        dotted blue (tot): all transitions, coincides with electronic case.
    '''
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    set_axes(ax)
    ax.set_xlim(-0.1, 4.2)
    if R < 5*Units.ANGSTROM2BOHR:
        ax.set_ylim(1e-2,1e3)
    else:
        ax.set_ylim(3*1e-4,60)
        
    plot_xs_el(ax, system, R, label="elec.", zorder=1)   
    L = icec.Morse_Dp.box_length
    plot_xs_tot(ax, system, R, vi, L, label=r'tot', modifier='-FC', color='tab:blue', ls=':')
    plot_xs_bc(ax, system, R, vi, L, label=r'b-d', modifier='-FC', color='tab:blue', ls='--')
    plot_xs_bb(ax, system, R, vi, label=r'b-b', modifier='-FC', color='tab:blue')
    
    ax.legend(ncols=2)
    fname = DIR_PLOTS + f'{system}.xs-FC.v{vi}.R{str(round(R*Units.BOHR2ANGSTROM))}.L{str(round(L*Units.BOHR2ANGSTROM))}.pdf'
    plt.tight_layout(pad=0.5)
    fig.savefig(fname)
    
def bb_vD(system, icec: IntraICEC, R, vD_max):
    ''' 
    Generates plot: 
        ICEC cross section vs. incoming electron energy for different initial vibrational states of D.
        Only bound-bound transitions of D are considered. 
        Results from vibrationally resolved PI cross sections and Franck-Condon model.
    
    Legend
        red: vi = 0
        purple: vi = 1
        blue: vi = 2
    '''
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    set_axes(ax)
    
    plot_xs_el(ax, system, R)
        
    color = ['tab:red', 'tab:purple', 'tab:blue']
    for vi in range(0, vD_max+1):
        label = r'$v_i=$' + str(vi)
        plot_xs_bb(ax, system, R, vi, label, color=color[vi])
        plot_xs_bb(ax, system, R, vi, label+' FC', modifier='-FC', linestyle='--', color=color[vi])
    
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='dimgray', ls=':', zorder=1)    
    
    fname = DIR_PLOTS + f'{system}.xs-FC.bb.R{round(R*Units.BOHR2ANGSTROM)}.pdf'
    plt.tight_layout(pad=0.5)
    fig.savefig(fname)
    
# ===== BOLTZMANN =====   
    
def boltzmann(icec: IntraICEC, results, vD_max, t):
    'Boltzmann weighted sum of ICEC cross section'
    # add De to energy(vi) to get positive values which increases numerical stability
    # exponent is unitless
    norm = sum(np.exp(-(icec.Morse_D.energy(vi)+icec.Morse_D.De)/Constants.KB/t) 
                for vi in range(vD_max+1)
                )
    avg = sum(
        np.exp(-(icec.Morse_D.energy(vi)+icec.Morse_D.De)/Constants.KB/t) * results[:, vi+1] 
        for vi in range(vD_max+1)
        )
    return avg/norm
    
def boltzmann_bb_and_bc(system, icec: IntraICEC, R, T, vD_max):
    ''' 
    Generates plot: 
        ICEC cross section against incoming electron energy for different temperatures.
        Franck-Condon model
    
    Legend
        Lighter shades indicate higher temperatures. 
        solid:  bound-bound transitions of LiH        
        dashed: bound-dissociative transitions
    '''
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    set_axes(ax)
    ax.set_xlim(-0.1, 4.2)
    
    modifier = "-FC"
    results_bb = file_io.read_xs(system, R, modifier)
    L = icec.Morse_Dp.box_length
    results_bc = file_io.read_xs(system, R, modifier+'.bc', L)

    results = results_bc
    for col in range(1,results.shape[1]):
        results[:, col] += results_bb[:, col]
        
    plot_xs_el(ax, system, R)
            
    blues = plt.get_cmap("Blues_r")    
    for idx, t in enumerate(T):
        blue = blues(idx / (len(T) + 1 / len(T)))
        label = r'$T=$' + str(t) + r'$\,$K'
        
        xs = boltzmann(icec, results_bb, vD_max, t)
        ax.plot(results_bb[:,0], xs, color=blue, label=label)
        
        xs = boltzmann(icec, results_bc, vD_max, t)
        ax.plot(results_bc[:,0], xs, color=blue, ls='--')
        
    ax.legend()
    fname = DIR_PLOTS + f'{system}.boltzmann-FC.R{round(R*Units.BOHR2ANGSTROM)}.L{round(L*Units.BOHR2ANGSTROM)}.pdf'
    plt.tight_layout(pad=0.5)
    fig.savefig(fname)