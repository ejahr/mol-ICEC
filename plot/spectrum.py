import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.constants import Units
from config import DIR
from plot.config import set_rcParams

set_rcParams()

def set_axes(ax, differential=False):
    ax.set_yscale('log')
    ax.set_xlabel(r"$\epsilon\prime$ [eV]")
    if differential:
        ax.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}E$ [Mb/eV]")
    else:
        ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    
def read_results(system, electronE, R, modifier='', L=None):
    file_path = DIR + f"results/{system}.spectrum{modifier}.E{round(electronE*Units.HARTREE2EV)}.R{round(R*Units.BOHR2ANGSTROM)}"
    if L is not None:
        file_path += f'.L{round(L*Units.BOHR2ANGSTROM)}.txt'
    else:
        file_path += '.txt'
    results = np.loadtxt(file_path, comments='#')
    return results   

def plot_icec_el(ax, icec_el: ICEC, electronE, R, width=0.002, return_bar=False):
    energy_out = icec_el.electronE_f(electronE)
    xs = icec_el.xs(electronE, R)
    if return_bar:
        return ax.bar(energy_out*Units.HARTREE2EV, xs*Units.AU2MB, width=width, color='black', label='elec.')
    ax.bar(energy_out*Units.HARTREE2EV, xs*Units.AU2MB, width=width, color='black', label='elec.') 
    
def spectrum_idx_electronEf(vD):
    return 4*vD+2

def spectrum_idx_xs(vD):
    return 4*vD+3
    
def plot_spectrum_FC(system, R, electronE, vD_max=0, title=None, icec_el:ICEC=None,):
    results_FC = read_results(system, electronE, R, modifier='-FC')
    results_resolved = read_results(system, electronE, R)
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    ax.set_title(title)
    set_axes(ax)
    #ax.set_ylim(1e-5, 3)
    ax.set_ylim(1e-3, 50)
    ax.set_xlim(6.545, 7.265)
    
    if icec_el is not None:
        plot_icec_el(ax, icec_el, electronE, R, width=0.003)
        # dummy line to get correct alignment in legend
        ax.bar(6.6, 1, width=0.005, color='white', alpha=0, label=' ')
    
    color_FC = ['tab:blue', 'rebeccapurple', 'tab:red']
    color_resolved = ['lightskyblue', 'mediumpurple', 'lightcoral']
    for vD in range(vD_max+1):
        label = str(vD) #r'$v_i=$' + 
        ax.bar(results_resolved[:,spectrum_idx_electronEf(vD)], results_resolved[:,spectrum_idx_xs(vD)], width=0.006, color=color_resolved[vD], label=label)
        ax.bar(results_FC[:,spectrum_idx_electronEf(vD)], results_FC[:,spectrum_idx_xs(vD)], width=0.002, color=color_FC[vD], label='FC')

    ax.legend(ncols=4, fontsize='small', loc='upper center')
    fname = DIR + f"plots/{system}.spectrum-FC.E{round(electronE*Units.HARTREE2EV)}.R{round(R*Units.BOHR2ANGSTROM)}.pdf"
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_spectrum_bc(system, icec:IntraICEC, R, electronE, vD=0, icec_el:ICEC=None):
    L=icec.Morse_Dp.box_length
    results_bb = read_results(system, electronE, R, modifier='-FC')
    results_bc = read_results(system, electronE, R, modifier='-FC.bc', L=L)
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    set_axes(ax)
    #ax.set_ylim(5*1e-4, 1)
    ax.set_ylim(1e-3, 10)
    ax2 = ax.twinx()
    ax2.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}E$ [Mb/eV]", rotation=-90)
    ax2.set_yticks([])
    ax2.yaxis.set_label_coords(1.06, 0.5)

    ax.plot(results_bc[:,spectrum_idx_electronEf(vD)], results_bc[:,spectrum_idx_xs(vD)], color='tab:blue', ls='--', label='b-d')
    ax.bar(results_bb[:,spectrum_idx_electronEf(vD)], results_bb[:,spectrum_idx_xs(vD)], width=0.005, color='tab:blue', label='b-b')
    
    if icec_el is not None:
        plot_icec_el(ax, icec_el, electronE, R, width=0.005)
    
    #x_min = min(results_bc[:,spectrum_idx_electronEf(vD)]) + 0.17
    #x_min = 5.69
    x_min = 5.5
    x_max = max(results_bb[:,spectrum_idx_electronEf(vD)]) + 0.04
    ax.set_xlim(x_min, x_max)
    ax.hlines(icec.PR_xs_A(electronE)*Units.AU2MB, 0, 10, color='dimgray', ls=':', zorder=1)  
    ax.annotate(r'$\sigma_\text{PR}$', 
                (x_min, icec.PR_xs_A(electronE)*Units.AU2MB), 
                xytext=(-26,-1),
                textcoords='offset points', color='dimgray') 

    ax.legend(fontsize='small', loc='upper left')
    fname = DIR + f"plots/{system}.spectrum-FC.bc.v0.E{round(electronE*Units.HARTREE2EV)}.R{round(R*Units.BOHR2ANGSTROM)}.L{round(L*Units.BOHR2ANGSTROM)}.pdf"
    plt.tight_layout()
    fig.savefig(fname)
    
def lorentzian(x, x0, gamma):
    # Cauchy, Lorentz, Breit-Wigner distribution
    # x0 : position of the peak
    # gamma : HWHM, FWHM = 2 * gamma, has units of x
    return (gamma / np.pi) / ((x - x0)**2 + gamma**2)
    #return gamma**2 / ((x - x0)**2 + gamma**2) # peak height stays the same
    
def boltzmann_bb(ax, icec: IntraICEC, results, vD_max, t, color, electronE=1*Units.EV2HARTREE, fold_lorentz=False):
    norm = icec.Morse_D.boltzmann_norm(t)
    
    if fold_lorentz:
        lorentzian_energies = np.linspace(results[-1,spectrum_idx_electronEf(0)]-1.5, results[0,spectrum_idx_electronEf(vD_max)]+1, 10000)
        lorentzian_spectrum = np.zeros_like(lorentzian_energies)
    
    for vD in range(vD_max+1):
        min_energy = icec.electronE_f_bc(electronE, vD, 0)*Units.HARTREE2EV
        occupation = icec.Morse_D.boltzmann_occupation(t, vD, norm=norm)
        energies = results[:,spectrum_idx_electronEf(vD)]
        spectrum = results[:,spectrum_idx_xs(vD)]
        if fold_lorentz:
            gamma = 0.08 #eV
            for energy, xs in zip(energies, spectrum):
                broadened_peak = xs * occupation * lorentzian(lorentzian_energies, energy, gamma)
                broadened_peak[lorentzian_energies < min_energy] = 0
                lorentzian_spectrum += broadened_peak 
                #ax.plot(lorentzian_energies, lorentzian_spectrum)
        else:
            ax.bar(energies, spectrum*occupation, width=0.005, color=color)
     
    if fold_lorentz:
        lorentzian_spectrum[lorentzian_spectrum<1e-5] = np.nan
        ax.plot(lorentzian_energies, lorentzian_spectrum, color=color, label = r'$T=$'+str(t)+r'$\,\mathrm{K}$')   
    
def interpolate(x0, x, y):
    interpolate_y = sp.interpolate.interp1d(
        x, y, kind="linear", bounds_error=False, fill_value=0
    )
    return interpolate_y(x0)
      
def boltzmann_bc(ax, icec: IntraICEC, results, vD_max, t, color):
    norm = icec.Morse_D.boltzmann_norm(t)
    
    energy = np.sort(
        np.concatenate(
            ([results[:,spectrum_idx_electronEf(vD)] for vD in range(vD_max+1)]), 
            axis=None
        )
    )
    
    energy_v0 = results[:,spectrum_idx_electronEf(0)]
    xs_v0 = results[:,spectrum_idx_xs(0)]
    xs_interpolated = interpolate(energy, energy_v0, xs_v0)
    avg = xs_interpolated * icec.Morse_D.boltzmann_occupation(t, 0, norm=norm)
    
    plot_all = False
    if plot_all:
        reds = plt.get_cmap("Reds_r")  
        if t>1000:
            ax.plot(energy_v0, xs_v0 * icec.Morse_D.boltzmann_occupation(t, 0, norm=norm), color=reds(0 / vD_max))
        
    for vD in range(1, vD_max+1):
        energy_vD = results[:,spectrum_idx_electronEf(vD)]
        xs_vD = results[:,spectrum_idx_xs(vD)]
        xs_interpolated = interpolate(energy, energy_vD, xs_vD)
        avg += xs_interpolated * icec.Morse_D.boltzmann_occupation(t, vD, norm=norm)
        
        if plot_all:
            if t>1000:
                ax.plot(energy_vD, xs_vD * icec.Morse_D.boltzmann_occupation(t, vD, norm=norm), color=reds(vD / vD_max))
    
    #avg[avg<1e-30]=np.nan
    ax.plot(energy, avg, color=color, ls="--", zorder=1)
    
def plot_boltzmann_FC(system, icec:IntraICEC, R, electronE, T, vD_max, icec_el:ICEC=None):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca() 
    set_axes(ax, differential=True)
    #ax.set_ylim(5*1e-4, 1)
    ax.set_ylim(1e-3, 10)
    ax.set_xlim(5.5, 8)
    
    L=icec.Morse_Dp.box_length
    results_bb_FC = read_results(system, electronE, R, modifier='-FC')
    results_bc_FC = read_results(system, electronE, R, modifier='-FC.bc', L=L)

    blues = plt.get_cmap("Blues_r")    
    for t in T:
        blue = blues(T.index(t) / (len(T) + 2 / len(T)))
        boltzmann_bb(ax, icec, results_bb_FC, vD_max, t, blue, electronE, fold_lorentz=True)
        boltzmann_bc(ax, icec, results_bc_FC, vD_max, t, blue)
        
    ax.legend(fontsize='small', loc="upper right")
    plt.tight_layout()
    fname = DIR + f'plots/{system}.boltzmann-FC.spectrum.R{round(R*Units.BOHR2ANGSTROM)}.L{round(L*Units.BOHR2ANGSTROM)}.pdf'
    plt.tight_layout()
    fig.savefig(fname)


