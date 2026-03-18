import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.constants import Units, Constants
from config import DIR
from plot.config import set_rcParams

set_rcParams()
width, height = 6, 4

def set_axes(ax):
    ax.set_yscale('log')
    ax.set_xlabel(r'$\varepsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)

def read_results_file(system, R, modifier='', L=None):
    file_path = DIR + f"results/{system}.xs{modifier}.R{round(R*Units.BOHR2ANGSTROM)}"
    if L is not None:
        file_path += f'.L{round(L*Units.BOHR2ANGSTROM)}.txt'
    else:
        file_path += '.txt'
    results = np.loadtxt(file_path, comments='#')
    return results

def plot_xs_vB_vBp(system, icec: IntraICEC, R, vD_max, vDp_max):
    fname = DIR + f"plots/{system}.all_vib.R{round(R*Units.BOHR2ANGSTROM)}.pdf"
    with PdfPages(fname) as pdf:
        energies = icec.energyGrid * Units.HARTREE2EV
        for vi in range(vD_max+1): 
            fig, ax = plt.subplots()
            set_axes(ax)
            ax.set_xlim(-0.2, 8.5)
            ax.set_ylim(1e-5, 1e2)
            results = read_results_file(system, R)
            ax.plot(results[:,0], results[:, vi+1], label='total', color='dimgray')
            for vf in range(vDp_max+1):
                label = r'$v_{LiH^+}=$' + str(vf)
                xs = icec.xs_vD_vDp(R, vi, vf) * Units.AU2MB
                ax.plot(energies, xs, label=label)
            icec.plot_PR_xs_A(ax, label=r'$\sigma_\text{PR}$', color='dimgray', ls=':', zorder=1)
            ax.legend()
            pdf.savefig(fig)  #, bbox_inches = "tight"
            plt.close(fig) 
            
def plot_xs_el(ax, icec:ICEC, R, label='electronic', color='black', **kwargs):
    energy = icec.energyGrid * Units.HARTREE2EV
    xs = icec.xs_energy(R) * Units.AU2MB
    ax.plot(energy, xs, color=color, label=label, **kwargs) #r'$R^{\mathrm{LiH}}_e$'

def plot_xs(ax, system, R, vD, label='icec', modifier='', **kwargs):
    if modifier == '':
      ax.set_xlim(-0.2, 8.6)  
    # ax.plot(icec.energyGrid * Units.HARTREE2EV,  icec.PI_xs_B(v_B, 0, icec.energyGrid + icec.IP_A)*Units.AU2MB, label=r'$\sigma_\text{PI}$')
    results = read_results_file(system, R, modifier)
    ax.plot(results[:,0], results[:, vD+1], label=label, **kwargs)
    
def plot_xs_bc(ax, system, R, vD, L, label='icec', modifier='', **kwargs):
    modifier += ".bc"
    results = read_results_file(system, R, modifier, L)
    ax.plot(results[:,0], results[:, vD+1], label=label, **kwargs)
    
def plot_xs_tot(ax, system, R, vD, L, label='icec', modifier='', **kwargs):
    results_bb = read_results_file(system, R, modifier)
    modifier += ".bc"
    results_bc = read_results_file(system, R, modifier, L)
    results = results_bb[:, vD+1] + results_bc[:, vD+1]
    ax.plot(results_bb[:,0], results, label=label, **kwargs)
    
def calculate_ratio_tot_vs_electronic(system, icec_el:ICEC, R, vD=0, L=8*Units.ANGSTROM2BOHR, modifier='-FC'):
    results_bb = read_results_file(system, R, modifier)
    modifier += ".bc"
    results_bc = read_results_file(system, R, modifier, L)
    results = results_bb[:, vD+1] + results_bc[:, vD+1]
    
    print("\n--- Ratio between total and electronic cross section ---")
    
    for i in [0,600,950]:
        energy = results_bb[i,0]
        xs_tot = results[i]
        xs_el = icec_el.xs(energy*Units.EV2HARTREE, R) * Units.AU2MB
        print(f"electronE    : {round(energy,3)} eV")
        print(f"xs_tot       : {round(xs_tot,3)} MB")
        print(f"xs_el        : {round(xs_el,3)} MB")
        print(f"xs_tot/xs_el : {round(xs_tot/xs_el,5)}")
    
def plot_xs_FC_bb(system, icec: IntraICEC, R, icec_el:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    set_axes(ax)
    if R < 5*Units.ANGSTROM2BOHR:
        ax.set_ylim(1e-4,1e3)
    if icec_el is not None:
        plot_xs_el(ax, icec_el, R)
    vi = 0
    plot_xs(ax, system, R, vi, label=r'b-b FC', modifier='-FC', color='tab:blue')
    plot_xs(ax, system, R, vi, label=r'b-b', color='tab:red')
    icec.plot_PR_xs_A(ax, label=r'$\sigma_\text{PR}$', color='dimgray', ls=':') 
    ax.legend()
    fname = DIR + f'plots/{system}.xs-FC.v0.R{round(R*Units.BOHR2ANGSTROM)}.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_xs_FC(system, icec: IntraICEC, R, icec_el:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    set_axes(ax)
    ax.set_xlim(-0.1, 4.2)
    if R < 5*Units.ANGSTROM2BOHR:
        ax.set_ylim(1e-2,1e3)
    else:
        ax.set_ylim(3*1e-4,60)
        
    if icec_el is not None:
        plot_xs_el(ax, icec_el, R, label="elec.", zorder=1)
    #icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='dimgray', ls=':')      
        
    vi = 0
    L = icec.Morse_Dp.box_length
    plot_xs_tot(ax, system, R, vi, L, label=r'tot', modifier='-FC', color='tab:blue', ls=':')
    plot_xs_bc(ax, system, R, vi, L, label=r'b-d', modifier='-FC', color='tab:blue', ls='--')
    plot_xs(ax, system, R, vi, label=r'b-b', modifier='-FC', color='tab:blue')
    #plot_xs(ax, system, R, vi, label=r'b-b', color='tab:blue') 
    
    ax.legend(ncols=2)
    fname = DIR + f'plots/{system}.xs-FC.bc.v0.R{str(round(R*Units.BOHR2ANGSTROM))}.L{str(round(L*Units.BOHR2ANGSTROM))}.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_xs_vi(system, icec: IntraICEC, R, vD_max, icec_el:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    set_axes(ax)
    
    if icec_el is not None:
        plot_xs_el(ax, icec_el, R)
        
    color = ['tab:red', 'tab:purple', 'tab:blue']
    for vi in range(0, vD_max+1):
        label = r'$v_i=$' + str(vi)
        plot_xs(ax, system, R, vi, label, color=color[vi])
        
    icec.plot_PR_xs_A(ax, label=r'$\sigma_\text{PR}$', color='dimgray', ls=':', zorder=1)
    
    ax.legend()
    fname = DIR + f'plots/{system}.vB.R{round(R*Units.BOHR2ANGSTROM)}.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_xs_vi_FC(system, icec: IntraICEC, R, vD_max, icec_el:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    set_axes(ax)
    
    if icec_el is not None:
        plot_xs_el(ax, icec_el, R)
        
    color = ['tab:red', 'tab:purple', 'tab:blue']
    for vi in range(0, vD_max+1):
        label = r'$v_i=$' + str(vi)
        plot_xs(ax, system, R, vi, label, color=color[vi])
        plot_xs(ax, system, R, vi, label+' FC', modifier='-FC', linestyle='--', color=color[vi])
    
    icec.plot_PR_xs_A(ax, label=r'$\sigma_\text{PR}$', color='dimgray', ls=':', zorder=1)    
    
    fname = DIR + f'plots/{system}.xs-FC.vB.R{round(R*Units.BOHR2ANGSTROM)}.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
    
def boltzmann(icec: IntraICEC, results, vD_max, t):
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
    
    
def plot_xs_boltzmann(system, icec: IntraICEC, R, T, vD_max, vib_energies=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    set_axes(ax)
    ax.set_xlim(-0.2, 8.5)
    results = read_results_file(system, R)
    
    blues = plt.get_cmap("Blues_r")    
    for t in T:
        if vib_energies is None:
            xs = boltzmann(icec, results, vD_max, t)
        else:
            norm = sum(
                np.exp(-vib_energies[vi]/Constants.KB/t) 
                for vi in range(vD_max+1)
                )
            avg = sum(
                np.exp(-vib_energies[vi]/Constants.KB/t) * results[:, vi+1]
                for vi in range(vD_max+1)
                )
            xs = avg/norm
        label = r'$T=$' + str(t) + 'K'
        blue = blues(T.index(t) / (len(T) + 2 / len(T)))
        ax.plot(results[:,0], xs, label=label, color=blue)
        
    #for vi in range(v_max + 1):
    #    plot_xs(ax, system, R, vi, r'$v_{LiH}=$'+str(vi), linestyle=':')  
        
    ax.legend()
    plt.tight_layout()
    fname = DIR + f'plots/{system}.boltzmann.R{round(R*Units.BOHR2ANGSTROM)}.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_xs_boltzmann_FC(system, icec: IntraICEC, R, T, vD_max, icec_el:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    set_axes(ax)
    ax.set_xlim(-0.1, 4.2)
    
    modifier = "-FC"
    results_bb = read_results_file(system, R, modifier)
    L = icec.Morse_Dp.box_length
    results_bc = read_results_file(system, R, modifier+'.bc', L)

    results = results_bc
    for col in range(1,results.shape[1]):
        results[:, col] += results_bb[:, col]
        
    if icec_el is not None:
        plot_xs_el(ax, icec_el, R)
            
    blues = plt.get_cmap("Blues_r")    
    for t in T:
        blue = blues(T.index(t) / (len(T) + 1 / len(T)))
        label = r'$T=$' + str(t) + r'$\,$K'
        
        xs = boltzmann(icec, results_bb, vD_max, t)
        ax.plot(results_bb[:,0], xs, color=blue, label=label)
        
        xs = boltzmann(icec, results_bc, vD_max, t)
        ax.plot(results_bc[:,0], xs, color=blue, ls='--')
        
        #xs = boltzmann(icec, results, vD_max, t)
        #ax.plot(results[:,0], xs, color=blue, ls=':')
    
    ax.legend()
    plt.tight_layout()
    fname = DIR + f'plots/{system}.boltzmann-FC.R{round(R*Units.BOHR2ANGSTROM)}.L{round(L*Units.BOHR2ANGSTROM)}.pdf'
    plt.tight_layout()
    fig.savefig(fname)

def plot_xs_R(system, icec: IntraICEC, R, icec_el:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    set_axes(ax)

    blues = plt.get_cmap("Blues_r")
    for r in R:
        index = np.where(R==r)[0][0]
        blue = blues(index / (len(R) + 1 / len(R)))
        label = r'$R=$' + str(round(r*Units.BOHR2ANGSTROM)) + r'$\,\mathrm{\AA}$'
        
        if icec_el is not None:
            plot_xs_el(ax, icec_el, r, color=blue, ls='--')
                
        plot_xs(ax, system, r, 0, label, color=blue)
    
    icec.plot_PR_xs_A(ax, label=r'$\sigma_\text{PR}$', color='dimgray', ls=':', zorder=1)
    plt.legend()
    fname = DIR + f'plots/{system}.R.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
