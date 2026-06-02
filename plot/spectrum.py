''' 
Defines functions for generating spectrum plots:
- bb_resolved_and_FC: ICEC cross section vs. outgoing electron energy for different initial vibrational states
- bb_and_bc: ICEC cross section vs. outgoing electron energy, includes bound-dissociative transitions of D
- boltzmann_bb_and_bc: ICEC cross section vs. outgoing electron energy for different temperatures.
'''

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.constants import Units
from config import DIR_PLOTS
from plot.config import set_rcParams
import calc.file_io as file_io

set_rcParams()

# ===== HELPER FUNCTIONS =====

def set_axes(ax, differential=False):
    ax.set_yscale('log')
    ax.set_xlabel(r"$\epsilon\prime$ [eV]")
    if differential:
        ax.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}E$ [Mb/eV]")
    else:
        ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
        
def interpolate(x0, x, y):
    interpolate_y = sp.interpolate.interp1d(
        x, y, kind="linear", bounds_error=False, fill_value=0
    )
    return interpolate_y(x0)

def diss_energy_secax(ax, vD, results_bc, label=r"$E_+$ [eV]"):
    electronEf = results_bc[:, file_io.spectrum_idx(vD, 'E_out')]
    diss_energy = results_bc[:, file_io.spectrum_idx(vD, 'diss_energy')]
    E_max = electronEf[0] + diss_energy[0]
    
    def electron_to_vib(electronEf):
        return E_max - electronEf
    
    def vib_to_electron(diss_energy):
        return E_max - diss_energy
    
    secax = ax.secondary_xaxis(
        'top',
        functions=(electron_to_vib, vib_to_electron)
    )
    secax.set_xlabel(label, labelpad = 8)
    secax.tick_params(axis='both', which='major', labelsize=14)
    
def lorentzian(x, x0, gamma):
    '''Cauchy, Lorentz, Breit-Wigner distribution
    x0 : position of the peak
    gamma : HWHM, FWHM = 2 * gamma, has units of x 
    '''
    return (gamma / np.pi) / ((x - x0)**2 + gamma**2)
    #return gamma**2 / ((x - x0)**2 + gamma**2) # peak height stays the same

# ===== SPECTRUM PLOTS =====

def plot_icec_el(ax, icec_el: ICEC, electronE, R, width=0.002, return_bar=False):
    energy_out = icec_el.electronE_f(electronE) * Units.HARTREE2EV
    xs = icec_el.xs(electronE, R) * Units.AU2MB
    if return_bar:
        return ax.bar(energy_out, xs, width=width, color='black', label='elec.')
    ax.bar(energy_out, xs, width=width, color='black', label='elec.') 
    
def bb_resolved_and_FC(system, R, electronE, vD_max=0, title=None, icec_el:ICEC=None,):
    ''' Generates spectrum plot: ICEC cross section vs. outgoing electron energy for different initial vibrational states.
    
    Legend
        Lighter shades (wider lines) include vibrationally resolved photoionization cross sections.
        Darker shades use the Franck-Condon model.
        Electronic case (black) corresponds to the vertical ionization of D.
    '''
    results_FC = file_io.read_spectrum(system, electronE, R, modifier='-FC.bb')
    results_resolved = file_io.read_spectrum(system, electronE, R, modifier='.bb')
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    ax.set_title(title)
    set_axes(ax)
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
        ax.bar(
            results_resolved[:,file_io.spectrum_idx(vD, 'E_out')], 
            results_resolved[:,file_io.spectrum_idx(vD, 'xs')], 
            width=0.006, color=color_resolved[vD], label=label)
        ax.bar(
            results_FC[:,file_io.spectrum_idx(vD, 'E_out')], 
            results_FC[:,file_io.spectrum_idx(vD, 'xs')], 
            width=0.002, color=color_FC[vD], label='FC')

    ax.legend(ncols=4, fontsize='small', loc='upper center')
    fname = os.path.join(
        DIR_PLOTS,
        f"{system}.spectrum.bb.E{round(electronE*Units.HARTREE2EV)}.R{round(R*Units.BOHR2ANGSTROM)}.pdf"
    )
    plt.tight_layout(pad=0.5)
    fig.savefig(fname)
    
def bb_and_bc(system, icec:IntraICEC, R, electronE, vD=0, icec_el:ICEC=None, secax_label=r'$E$ [eV]'):
    ''' Generates spectrum plot: ICEC cross section vs. outgoing electron energy.
    
    Legend
        Blue peaks (b-b): bound-bound transitions during ionization of D.
        Blue dashed line (b-d): differential dsigma/dE for bound-dissociative transitions.
        Black peak: electronic case (vertical ionization of D).
    '''
    L=icec.Morse_Dp.box_length
    results_bb = file_io.read_spectrum(system, electronE, R, modifier='-FC.bb')
    results_bc = file_io.read_spectrum(system, electronE, R, modifier='-FC.bc', L=L)
    
    fig = plt.figure(figsize=(6,4.1))
    ax = plt.gca() 
    set_axes(ax)
    ax.set_ylim(1e-3, 10)
    ax2 = ax.twinx()
    ax2.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}E$ [Mb/eV]", rotation=-90)
    ax2.set_yticks([])
    ax2.yaxis.set_label_coords(1.06, 0.5)

    ax.plot(
        results_bc[:,file_io.spectrum_idx(vD, 'E_out')], 
        results_bc[:,file_io.spectrum_idx(vD, 'xs')], 
        color='tab:blue', ls='--', label='b-d')
    ax.bar(
        results_bb[:,file_io.spectrum_idx(vD, 'E_out')], 
        results_bb[:,file_io.spectrum_idx(vD, 'xs')], 
        width=0.005, color='tab:blue', label='b-b')
    
    if icec_el is not None:
        plot_icec_el(ax, icec_el, electronE, R, width=0.005)
    
    x_min = 5.5
    x_max = max(results_bb[:,file_io.spectrum_idx(vD, 'E_out')]) + 0.04
    ax.set_xlim(x_min, x_max)
    ax.hlines(icec.PR_xs_A(electronE)*Units.AU2MB, 0, 10, color='dimgray', ls=':', zorder=1)  
    ax.annotate(r'$\sigma_\text{PR}$', 
                (x_min, icec.PR_xs_A(electronE)*Units.AU2MB), 
                xytext=(-26,-1),
                textcoords='offset points', color='dimgray') 
    
    #print('PR xs =', icec.PR_xs_A(electronE)*Units.AU2MB, 'eV')
    
    diss_energy_secax(ax, vD, results_bc, label=secax_label)

    ax.legend(fontsize='small', loc='upper left')
    fname = os.path.join(
        DIR_PLOTS,
        f"{system}.spectrum-FC.v0.E{round(electronE*Units.HARTREE2EV)}.R{round(R*Units.BOHR2ANGSTROM)}.L{round(L*Units.BOHR2ANGSTROM)}.pdf"
    )
    plt.tight_layout(pad = 0.5)
    fig.savefig(fname)
    
# ===== TEMPERATURE DEPENDENT SPECTRUM PLOTS ======
    
def plot_boltzmann_bb(ax, icec: IntraICEC, results, vD_max, t, color, electronE=1*Units.EV2HARTREE, fold_lorentz=False, **kwargs):
    norm = icec.Morse_D.boltzmann_norm(t)
    
    if fold_lorentz:
        lorentzian_energies = np.linspace(results[-1,file_io.spectrum_idx(0, 'E_out')]-1.5, results[0,file_io.spectrum_idx(vD_max,'E_out')]+1, 10000)
        lorentzian_spectrum = np.zeros_like(lorentzian_energies)
    
    for vD in range(vD_max+1):
        min_energy = icec.electronE_f_bc(electronE, vD, 0)*Units.HARTREE2EV
        occupation = icec.Morse_D.boltzmann_occupation(t, vD, norm=norm)
        energies = results[:,file_io.spectrum_idx(vD, 'E_out')]
        spectrum = results[:,file_io.spectrum_idx(vD, 'xs')]
        if fold_lorentz:
            gamma = 0.08 #eV
            for energy, xs in zip(energies, spectrum):
                broadened_peak = xs * occupation * lorentzian(lorentzian_energies, energy, gamma)
                broadened_peak[lorentzian_energies < min_energy] = 0
                lorentzian_spectrum += broadened_peak 
        else:
            ax.bar(energies, spectrum*occupation, width=0.005, color=color, **kwargs)
     
    if fold_lorentz:
        lorentzian_spectrum[lorentzian_spectrum<1e-5] = np.nan
        ax.plot(lorentzian_energies, lorentzian_spectrum, color=color, label = r'$T=$'+str(t)+r'$\,\mathrm{K}$', **kwargs)   
      
def plot_boltzmann_bc(ax, icec: IntraICEC, results, vD_max, t, color, **kwargs):
    norm = icec.Morse_D.boltzmann_norm(t)
    
    energy = np.sort(
        np.concatenate(
            ([results[:,file_io.spectrum_idx(vD, 'E_out')] for vD in range(vD_max+1)]), 
            axis=None
        )
    )
    
    energy_v0 = results[:,file_io.spectrum_idx(0, 'E_out')]
    xs_v0 = results[:,file_io.spectrum_idx(0, 'xs')]
    xs_interpolated = interpolate(energy, energy_v0, xs_v0)
    avg = xs_interpolated * icec.Morse_D.boltzmann_occupation(t, 0, norm=norm)
        
    for vD in range(1, vD_max+1):
        energy_vD = results[:,file_io.spectrum_idx(vD, 'E_out')]
        xs_vD = results[:,file_io.spectrum_idx(vD, 'xs')]
        xs_interpolated = interpolate(energy, energy_vD, xs_vD)
        avg += xs_interpolated * icec.Morse_D.boltzmann_occupation(t, vD, norm=norm)
    
    ax.plot(energy, avg, color=color, ls="--", **kwargs)
    
def boltzmann_bb_and_bc(system, icec:IntraICEC, R, electronE, T, vD_max, icec_el:ICEC=None):
    ''' Generates spectrum plot: ICEC cross section against outgoing electron energy for different temperatures.
    
    Legend
        Lighter shades indicate higher temperatures. 
        solid:  bound-bound transitions of LiH, folded with a Lorentz distribution
        dashed: bound-dissociative transitions
    '''
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca() 
    set_axes(ax, differential=True)

    ax.set_ylim(1e-3, 10)
    ax.set_xlim(5.5, 8)
    
    L=icec.Morse_Dp.box_length
    results_bb_FC = file_io.read_spectrum(system, electronE, R, modifier='-FC.bb')
    results_bc_FC = file_io.read_spectrum(system, electronE, R, modifier='-FC.bc', L=L)

    blues = plt.get_cmap("Blues_r")  
    if len(T) == 3:
        blues = [blues(0.01), blues(0.3), blues(0.55)]
    else:
        blues = [blues(idx / (len(T) + 2 / len(T))) for idx in range(len(T))]
    zorders = [len(T)-i for i in range(len(T))]
    for t, zorder, blue in zip(T, zorders, blues):
        plot_boltzmann_bc(ax, icec, results_bc_FC, vD_max, t, blue, zorder=zorder)
        plot_boltzmann_bb(ax, icec, results_bb_FC, vD_max, t, blue, electronE, fold_lorentz=True, zorder=zorder+len(T))
        
    ax.legend(fontsize='small', loc="upper right")
    fname = os.path.join(
        DIR_PLOTS,
        f'{system}.spectrum-FC.boltzmann.E{round(electronE*Units.HARTREE2EV)}.R{round(R*Units.BOHR2ANGSTROM)}.L{round(L*Units.BOHR2ANGSTROM)}.pdf'
    )
    plt.tight_layout(pad = 0.5)
    fig.savefig(fname)