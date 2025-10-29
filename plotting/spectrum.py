import numpy as np
import matplotlib.pyplot as plt
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.constants import Units
from plotting.base import DIR

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 16})

def set_axes(ax):
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon_\text{out}$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)

def plot_spectrum(system, icec:IntraICEC, R, electronE, vD_max=0, title=None, icec_fixed:ICEC=None, modifier=''):
    fname = DIR + "results/" + system + ".spectrum.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    results = np.loadtxt(fname, comments='#')
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    set_axes(ax)
    ax.set_title(title)
    
    ax.hlines(icec.PR_xs_A(electronE)*Units.AU2MB, 0, 10, color='gray', ls=':', zorder=0)   
    
    color = ['tab:red', 'tab:purple', 'tab:blue']

    bars = [None] * (vD_max+1)
    for vi in range(vD_max+1):
        label = r'$v_i=$' + str(vi)
        bars[vi] = ax.bar(results[:,3*vi], results[:,3*vi+1], width=0.002, label=label, color=color[vi])
    labels = [r'$v_i=$' + str(vi) for vi in range(vD_max+1)] 
        
    if icec_fixed is not None:
        energy_out = icec_fixed.electronE_f(electronE)
        xs = icec_fixed.xs(electronE, R)
        bars.append(ax.bar(energy_out*Units.HARTREE2EV, xs*Units.AU2MB, width=0.002, color='black', label='unresolved'))
    labels.append('unresolved')
        
    ax.legend(handles=[bar[0] for bar in bars], labels=labels, ncols=2, fontsize='small', loc='upper right')
    
    x_min = min(rect.get_x() for bar in bars for rect in bar)
    x_max = max(rect.get_x() for bar in bars for rect in bar)
    ax.set_xlim(x_min - 0.025, x_max + 0.025)
    
    ax.annotate(r'$\sigma_\text{PR}$', (x_max + 0.025, icec.PR_xs_A(electronE)*Units.AU2MB), xytext=(3,-3),    # fraction, fraction
            textcoords='offset points', color='gray')
    
    fname = DIR + 'plots/' + system + ".spectrum" + modifier + ".E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.pdf"
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_spectrum_FC(system, R, electronE, vD_max=0, title=None):
    fname = DIR + "results/" + system + ".spectrum-FC.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    results_FC = np.loadtxt(fname, comments='#')
    
    fname = DIR + "results/" + system + ".spectrum.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    results_resolved = np.loadtxt(fname, comments='#')
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    ax.set_title(title)
    set_axes(ax)
    ax.set_ylim(2*1e-4, 0.2)
    
    color_resolved = ['tab:red', 'tab:purple', 'tab:blue']
    color_FC = ['tab:orange', 'violet' ,'lightskyblue']
    for vi in range(vD_max+1):
        label = r'$v_i=$' + str(vi)
        ax.bar(results_resolved[:,3*vi], results_resolved[:,3*vi+1], width=0.004, color=color_resolved[vi], label=label)
        ax.bar(results_FC[:,3*vi], results_FC[:,3*vi+1], width=0.002, color=color_FC[vi], label=label + ' FC')

    ax.legend(ncols=3, fontsize='small', loc='upper center')
    fname = DIR + 'plots/' + system + ".spectrum-FC.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.pdf"
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_spectrum_bc(system, R, electronE, vi=0):
    fname = DIR + "results/" + system + ".spectrum-FC.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    results_bb = np.loadtxt(fname, comments='#')
    
    fname = DIR + "results/" + system + ".spectrum-FC.bc.v0.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    results_bc = np.loadtxt(fname, comments='#')
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    set_axes(ax)
    ax.set_ylim(2*1e-4, 20)

    ax.plot(results_bc[:,0], results_bc[:,1], color='tab:red', label='dissociation')
    ax.bar(results_bb[:,3*vi], results_bb[:,3*vi+1], width=0.002, color='tab:blue', label='bound')

    ax.legend(ncols=3, fontsize='small', loc='upper center')
    fname = DIR + 'plots/' + system + ".spectrum-FC.bc.v0.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.pdf"
    plt.tight_layout()
    fig.savefig(fname)