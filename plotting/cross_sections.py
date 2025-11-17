import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.constants import Units, Constants
from plotting.base import DIR

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 16})
width, height = 6, 4

def read_results_file(system, R, modifier='', L=None):
    file_path = DIR + f"results/{system}.xs{modifier}.R{str(round(R*Units.BOHR2ANGSTROM))}"
    if L is not None:
        file_path += f'.L{str(round(L*Units.BOHR2ANGSTROM))}.txt'
    else:
        file_path += 'icec.txt'
    results = np.loadtxt(file_path, comments='#')
    return results

def plot_xs_vB_vBp(system, icec: IntraICEC, R, vD_max, vDp_max):
    fname = DIR + "plots/" + system + '.all_vib.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.pdf"
    with PdfPages(fname) as pdf:
        energies = icec.energyGrid*Units.HARTREE2EV
        for vi in range(vD_max+1): 
            fig, ax = plt.subplots()
            ax.set_yscale('log')
            ax.set_xlabel(r'$\epsilon$ [eV]')
            ax.set_ylabel(r'$\sigma$ [Mb]')
            ax.set_xlim(-0.2, 8.5)
            ax.set_ylim(1e-5, 1e2)
            ax.grid(True)
            results = read_results_file(system, R)
            ax.plot(results[:,0], results[:, vi+1], label='total', color='grey')
            for vf in range(vDp_max+1):
                label = r'$v_{LiH^+}=$' + str(vf)
                xs = icec.xs_vD_vDp(R, vi, vf)
                ax.plot(energies, xs, label=label)
            icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', ls=':', zorder=0)
            ax.legend()
            pdf.savefig(fig)  #, bbox_inches = "tight"
            plt.close(fig) 

def plot_xs(ax, system, R, vD, label='icec', modifier='', **kwargs):
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    ax.set_xlim(-0.2, 8.6)
    # ax.plot(icec.energyGrid * Units.HARTREE2EV,  icec.PI_xs_B(v_B, 0, icec.energyGrid + icec.IP_A)*Units.AU2MB, label=r'$\sigma_\text{PI}$')
    results = read_results_file(system, R, modifier)
    ax.plot(results[:,0], results[:, vD+1], label=label, **kwargs)
    
def plot_xs_bc(ax, system, R, vD, L, label='icec', modifier='', **kwargs):
    modifier += ".bc"
    results = read_results_file(system, R, modifier, L)
    ax.plot(results[:,0], results[:, vD+1], label=label, **kwargs)
    
def plot_xs_FC(system, icec: IntraICEC, R, icec_fixed:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    if icec_fixed is not None:
        energy = icec_fixed.energyGrid*Units.HARTREE2EV
        xs = icec_fixed.xs_energy(R)
        ax.plot(energy, xs, color='gray', label=r'unresolved')
    vi = 0
    L = icec.Morse_Dp.box_length
    plot_xs_bc(ax, system, R, vi, L, label=r'dissociation', modifier='-FC', color='tab:red', ls='--')
    plot_xs(ax, system, R, vi, label=r'FC', modifier='-FC', color='tab:red')
    plot_xs(ax, system, R, vi, label=r'resolved', color='tab:blue')
    ax.legend()
    fname = DIR + f'plots/{system}.xs-FC.v0.R{str(round(R*Units.BOHR2ANGSTROM))}.L{str(round(L*Units.BOHR2ANGSTROM))}.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_xs_vi(system, icec: IntraICEC, R, vD_max, icec_fixed:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    if icec_fixed is not None:
        energy = icec_fixed.energyGrid*Units.HARTREE2EV
        xs = icec_fixed.xs_energy(R)
        ax.plot(energy, xs, color='gray', label='unresolved')
        
    color = ['tab:red', 'tab:purple', 'tab:blue']
    for vi in range(0, vD_max+1):
        label = r'$v_i=$' + str(vi)
        plot_xs(ax, system, R, vi, label, color=color[vi])
        
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', ls=':', zorder=0)
    
    ax.legend()
    fname = DIR + 'plots/' + system + '.vB.R'+ str(round(R*Units.BOHR2ANGSTROM)) + '.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_xs_vi_FC(system, icec: IntraICEC, R, vD_max, icec_fixed:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    
    if icec_fixed is not None:
        energy = icec_fixed.energyGrid*Units.HARTREE2EV
        xs = icec_fixed.xs_energy(R)
        ax.plot(energy, xs, color='gray', label='unresolved')
        
    color = ['tab:red', 'tab:purple', 'tab:blue']
    for vi in range(0, vD_max+1):
        label = r'$v_i=$' + str(vi)
        plot_xs(ax, system, R, vi, label, color=color[vi])
        plot_xs(ax, system, R, vi, label+' FC', modifier='-FC', linestyle='--', color=color[vi])
    
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', ls=':', zorder=0)    
    
    fname = DIR + 'plots/' + system + '.xs-FC.vB.R'+ str(round(R*Units.BOHR2ANGSTROM)) + '.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_xs_boltzmann(system, icec: IntraICEC, R, T, vD_max, vib_energies=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.set_yscale('log')
    ax.set_xlim(-0.2, 8.5)
    ax.grid(True)
    results = read_results_file(system, R)
    
    blues = plt.get_cmap("Blues_r")    
    for t in T:
        if vib_energies is None:
            # add De to energy(vi) to get positive values which increases numerical stability
            norm = sum(np.exp(-(icec.Morse_D.energy(vi)+icec.Morse_D.De)/Constants.KB/t) 
                       for vi in range(vD_max+1)
                       )
            avg = sum(
                np.exp(-(icec.Morse_D.energy(vi)+icec.Morse_D.De)/Constants.KB/t) * results[:, vi+1] 
                for vi in range(vD_max+1)
                )
        else:
            norm = sum(
                np.exp(-vib_energies[vi]/Constants.KB/t) 
                for vi in range(vD_max+1)
                )
            avg = sum(
                np.exp(-vib_energies[vi]/Constants.KB/t) * results[:, vi+1]
                for vi in range(vD_max+1)
                )
        label = r'$T=$' + str(t) + 'K'
        blue = blues(T.index(t) / (len(T) + 1 / len(T)))
        ax.plot(results[:,0], avg/norm, label=label, color=blue)
        
    #for vi in range(v_max + 1):
    #    plot_xs(ax, system, R, vi, r'$v_{LiH}=$'+str(vi), linestyle=':')  
        
    ax.legend()
    plt.tight_layout()
    fname = DIR + 'plots/' + system + '.boltzmann.R'+ str(round(R*Units.BOHR2ANGSTROM)) + '.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)

def plot_xs_R(system, icec: IntraICEC, R, icec_fixed:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 

    blues = plt.get_cmap("Blues_r")
    for r in R:
        blue = blues(R.index(r) / (len(R) + 1 / len(R)))
        label = r'$R=$' + str(round(r*Units.BOHR2ANGSTROM)) + r'$\,\mathrm{\AA}$'
        
        if icec_fixed is not None:
            energy = icec_fixed.energyGrid*Units.HARTREE2EV
            xs = icec_fixed.xs_energy(r)
            ax.plot(energy, xs, color=blue, ls='--')
                
        plot_xs(ax, system, r, 0, label, color=blue)
    
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', ls=':', zorder=0)
    plt.legend()
    fname = DIR + 'plots/' + system + '.R.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
