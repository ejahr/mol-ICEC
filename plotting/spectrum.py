import numpy as np
import matplotlib.pyplot as plt
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.constants import Units
from plotting.base import DIR, set_rcParams

set_rcParams()

def set_axes(ax):
    ax.set_yscale('log')
    ax.set_xlabel(r"$\epsilon\prime$ [eV]")
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    
# TODO remove icec in filename
def read_results_file(system, electronE, R, modifier='', L=None):
    file_path = DIR + f"results/{system}.spectrum{modifier}.E{str(round(electronE*Units.HARTREE2EV))}.R{str(round(R*Units.BOHR2ANGSTROM))}"
    if L is not None:
        file_path += f'.L{str(round(L*Units.BOHR2ANGSTROM))}.txt'
    else:
        file_path += '.icec.txt'
    results = np.loadtxt(file_path, comments='#')
    return results    
    
def plot_spectrum(system, icec:IntraICEC, R, electronE, vD_max=0, title=None, icec_el:ICEC=None, modifier=''):
    results = read_results_file(system, electronE, R)
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    set_axes(ax)
    ax.set_title(title)
    
    color = ['tab:blue', 'tab:purple', 'tab:red']

    bars = [None] * (vD_max+1)
    for vi in range(vD_max+1):
        label = r'$v_i=$' + str(vi)
        bars[vi] = ax.bar(results[:,3*vi], results[:,3*vi+1], width=0.002, label=label, color=color[vi])
    labels = [r'$v_i=$' + str(vi) for vi in range(vD_max+1)] 
        
    if icec_el is not None:
        energy_out = icec_el.electronE_f(electronE)
        xs = icec_el.xs(electronE, R)
        bars.append(ax.bar(energy_out*Units.HARTREE2EV, xs*Units.AU2MB, width=0.002, color='black', label='electronic'))
    labels.append('electronic')
        
    ax.legend(handles=[bar[0] for bar in bars], labels=labels, ncols=2, fontsize='small', loc='upper right')
    
    x_min = min(rect.get_x() for bar in bars for rect in bar)
    x_max = max(rect.get_x() for bar in bars for rect in bar)
    ax.set_xlim(x_min - 0.025, x_max + 0.025)
    
    ax.hlines(icec.PR_xs_A(electronE)*Units.AU2MB, 0, 10, color='dimgray', ls=':', zorder=0)   
    #ax.annotate(r'$\sigma_\text{PR}$', (x_max + 0.025, icec.PR_xs_A(electronE)*Units.AU2MB), xytext=(3,-3),    # fraction, fraction
    #        textcoords='offset points', color='dimgray')
    ax.annotate(r'$\sigma_\text{PR}$', 
            (x_min-0.025, icec.PR_xs_A(electronE)*Units.AU2MB), 
            xytext=(-26,-1),
            textcoords='offset points', color='dimgray') 
    
    fname = DIR + 'plots/' + system + ".spectrum" + modifier + ".E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.pdf"
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_spectrum_FC(system, R, electronE, vD_max=0, title=None, icec_el:ICEC=None,):
    results_FC = read_results_file(system, electronE, R, modifier='-FC')
    results_resolved = read_results_file(system, electronE, R)
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    ax.set_title(title)
    set_axes(ax)
    ax.set_ylim(1e-5, 3)
    ax.set_xlim(6.545, 7.265)
    
    if icec_el is not None:
        energy_out = icec_el.electronE_f(electronE)
        xs = icec_el.xs(electronE, R)
        ax.bar(energy_out*Units.HARTREE2EV, xs*Units.AU2MB, width=0.003, color='black', label='elec.')
        # dummy line to get correct alignment in legend
        ax.bar(6.6, 1, width=0.005, color='white', alpha=0, label=' ')
        
    
    color_FC = ['tab:blue', 'rebeccapurple', 'tab:red']
    color_resolved = ['lightskyblue', 'mediumpurple', 'lightcoral']
    for vi in range(vD_max+1):
        label = str(vi) #r'$v_i=$' + 
        ax.bar(results_resolved[:,3*vi], results_resolved[:,3*vi+1], width=0.006, color=color_resolved[vi], label=label)
        ax.bar(results_FC[:,3*vi], results_FC[:,3*vi+1], width=0.002, color=color_FC[vi], label='FC')

    ax.legend(ncols=4, fontsize='small', loc='upper center')
    fname = DIR + 'plots/' + system + ".spectrum-FC.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.pdf"
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_spectrum_bc(system, icec:IntraICEC, R, electronE, vi=0, icec_el:ICEC=None):
    L=icec.Morse_Dp.box_length
    #results_bb = read_results_file(system, electronE, R)
    results_bb_FC = read_results_file(system, electronE, R, modifier='-FC')
    results_bc_FC = read_results_file(system, electronE, R, modifier='-FC.bc.v0', L=L)
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    set_axes(ax)
    ax.set_ylim(5*1e-4, 1)
    ax2 = ax.twinx()
    ax2.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}E$ [Mb/eV]", rotation=-90)
    ax2.set_yticks([])
    ax2.yaxis.set_label_coords(1.06, 0.5)

    ax.plot(results_bc_FC[:,0], results_bc_FC[:,1], color='tab:blue', ls='--', label='b-d') # marker='.',
    ax.bar(results_bb_FC[:,3*vi], results_bb_FC[:,3*vi+1], width=0.005, color='tab:blue', label='b-b')
    #ax.bar(results_bb[:,3*vi], results_bb[:,3*vi+1], width=0.0075, color='tab:blue', label='b-b')
    
    if icec_el is not None:
        energy_out = icec_el.electronE_f(electronE)
        xs = icec_el.xs(electronE, R)
        ax.bar(energy_out*Units.HARTREE2EV, xs*Units.AU2MB, width=0.005, color='black', label='electronic')
    
    x_min = min(results_bc_FC[:,0]) + 0.17
    x_max = max(results_bb_FC[:,3*vi]) + 0.04
    ax.set_xlim(x_min, x_max)
    ax.hlines(icec.PR_xs_A(electronE)*Units.AU2MB, 0, 10, color='dimgray', ls=':', zorder=0)  
    ax.annotate(r'$\sigma_\text{PR}$', 
                (x_min, icec.PR_xs_A(electronE)*Units.AU2MB), 
                xytext=(-26,-1),
                textcoords='offset points', color='dimgray') 

    ax.legend(fontsize='small', loc='upper left')
    fname = DIR + f"plots/{system}.spectrum-FC.bc.v0.E{str(round(electronE*Units.HARTREE2EV))}.R{str(round(R*Units.BOHR2ANGSTROM))}.L{round(L*Units.BOHR2ANGSTROM)}.pdf"
    plt.tight_layout()
    fig.savefig(fname)