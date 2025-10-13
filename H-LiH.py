import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from input.HLiH import * 
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.constants import *

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 14})

DIR = '/home/elena/intraICEC/dimers/'

#https://doi.org/10.1021/jp9921295
R = 2 * ANGSTROM2BOHR
#R = 10 * ANGSTROM2BOHR
R = np.array([2,4,6,8,10]) * ANGSTROM2BOHR

min_kinE = 0.01 * EV2HARTREE
max_kinE = 10 * EV2HARTREE
resolution = 1000

electron_energies = np.array([1, 5]) * EV2HARTREE

def plot_xs_vB_vBp(system, icec: IntraICEC, R):
    fname = DIR + "plots/" + system + '.all_vib.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.pdf"
    with PdfPages(fname) as pdf:
        energies = icec.energyGrid*HARTREE2EV
        for vi in range(v_max+1): 
            fig, ax = plt.subplots()
            ax.set_yscale('log')
            ax.set_xlabel(r'$\epsilon$ [eV]')
            ax.set_ylabel(r'$\sigma$ [Mb]')
            ax.set_title(r'$\text{H}^+ \text{LiH}$, $v_{LiH}=$' + str(vi))
            ax.set_ylim(1e-5, 1e2)
            ax.grid(True)
            icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', linestyle= '--')
            results = read_results_file(system, R)
            ax.plot(results[:,0], results[:, vi+1], label='total', color='grey')
            for vf in range(vp_max+1):
                label = r'$v_{LiH^+}=$' + str(vf)
                xs = icec.xs_vD_vDp(R, vi, vf)
                ax.plot(energies, xs, label=label)
            ax.legend()
            pdf.savefig(fig)  #, bbox_inches = "tight"
            plt.close(fig) 
    

def calculate_xs_bb(system, header, icec: IntraICEC, R, v_max, vp_max, modifier=''):
    xs_array = icec.energyGrid*HARTREE2EV
    for v in range(v_max+1):
        xs = icec.xs_vD(R, v, vp_max)
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR + "results/" + system + '.xs' + modifier + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)
    
def calculate_xs_R(system, icec, R, header):
    for r in R:
        headerR = header + f'R = {round(r*BOHR2ANGSTROM)} Angstrom'
        headerR += 'E_in [eV] | xs [Mb]'
        calculate_xs_bb(system, headerR, icec, r, v_max, vp_max)

def calculate_spectrum(system, header, icec: IntraICEC, R, electronE, modifier=''): 
    new_header = header + "E_in = " + str(round(electronE*HARTREE2EV)) + " eV\n"
    new_header += "| E_out [eV] : xs [Mb] |"  
    spectrum_all_vi = np.array([]) 
    for vi in range(v_max+1):
        spectrum = icec.spectrum(electronE, R, vi, vp_max)
        if spectrum_all_vi.size == 0:
            spectrum_all_vi = spectrum
        else:
            spectrum_all_vi = np.hstack((spectrum_all_vi, spectrum))          
    fname = DIR + "results/" + system + ".spectrum" + modifier + ".E"+ str(round(electronE*HARTREE2EV)) + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.txt"
    np.savetxt(fname, spectrum_all_vi, fmt='%1.3e', header=new_header)  
    
def read_results_file(system, R, modifier=''):
    file_path = DIR + "results/" + system + '.xs' + modifier + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.txt"
    results = np.loadtxt(file_path, comments='#')
    return results

def plot_xs(ax, system, R, v_B, label='icec', modifier='', **kwargs):
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    # ax.plot(icec.energyGrid * HARTREE2EV,  icec.PI_xs_B(v_B, 0, icec.energyGrid + icec.IP_A)*AU2MB, label=r'$\sigma_\text{PI}$')
    results = read_results_file(system, R, modifier=modifier)
    ax.plot(results[:,0], results[:, v_B+1], label=label, **kwargs)
    ax.legend()
    
def plot_xs_FC(system, icec: IntraICEC, R, icec_fixed:ICEC=None):
    fig = plt.figure()
    ax = plt.gca() 
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', linestyle= '--')
    if icec_fixed is not None:
        energy = icec_fixed.energyGrid*HARTREE2EV
        xs = icec_fixed.xs_energy(R)
        ax.plot(energy, xs, color='gray', label=r'unresolved')
    vi = 0
    plot_xs(ax, system, R, vi, label=r'FC', modifier='-FC', color='tab:red')
    plot_xs(ax, system, R, vi, label=r'resolved', color='tab:blue')
    fname = DIR + 'plots/' + system + '.xs-FC.v0.R'+ str(round(R*BOHR2ANGSTROM)) + '.icec.pdf'
    fig.savefig(fname)
    
def plot_xs_vi(system, icec: IntraICEC, R):
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_title('ICEC cross section ' + r'$\text{H}^+ \text{LiH}$')
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', linestyle= '--')
    for vi in range(0, v_max+1):
        label = r'$v_{LiH}=$' + str(vi)
        plot_xs(ax, system, R, vi, label)
    fname = DIR + 'plots/' + system + '.vB.R'+ str(round(R*BOHR2ANGSTROM)) + '.icec.pdf'
    fig.savefig(fname)
    
def plot_xs_vi_FC(system, icec: IntraICEC, R, icec_fixed:ICEC=None):
    fig = plt.figure()
    ax = plt.gca() 
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', linestyle= '--')
    if icec_fixed is not None:
        energy = icec_fixed.energyGrid*HARTREE2EV
        xs = icec_fixed.xs_energy(R)
        ax.plot(energy, xs, color='gray', label='unresolved')
    color = ['red', 'violet', 'blue']
    for vi in range(0, v_max+1):
        label = r'$v_i=$' + str(vi)
        plot_xs(ax, system, R, vi, label, color=color[vi])
        plot_xs(ax, system, R, vi, label+' FC', modifier='-FC', linestyle='--', color=color[vi])
    fname = DIR + 'plots/' + system + '.xs-FC.vB.R'+ str(round(R*BOHR2ANGSTROM)) + '.icec.pdf'
    fig.savefig(fname)
    
    
def plot_xs_boltzmann(system, icec: IntraICEC, R, T, vib_energies=None):
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.set_yscale('log')
    ax.grid(True)
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', linestyle= '--')
    results = read_results_file(system, R)
    
    for t in T:
        if vib_energies is None:
            norm = sum(np.exp(-icec.Morse_D.energy(vi)/KB/t) 
                       for vi in range(v_max+1)
                       )
            avg = sum(
                np.exp(-icec.Morse_D.energy(vi)/KB/t) * results[:, vi+1] 
                for vi in range(v_max+1)
                )
        else:
            norm = sum(
                np.exp(-vib_energies[vi]/KB/t) 
                for vi in range(v_max+1)
                )
            avg = sum(
                np.exp(-vib_energies[vi]/KB/t) * results[:, vi+1]
                for vi in range(v_max+1)
                )
        label = r'$T=$' + str(t) + 'K'
        ax.plot(results[:,0], avg/norm, label=label)
        
    #for vi in range(v_max + 1):
    #    plot_xs(ax, system, R, vi, r'$v_{LiH}=$'+str(vi), linestyle=':')  
        
    ax.legend()
    plt.tight_layout()
    fname = DIR + 'plots/' + system + '.boltzmann.R'+ str(round(R*BOHR2ANGSTROM)) + '.icec.pdf'
    fig.savefig(fname)
    
def plot_xs_R(system, icec: IntraICEC, R):
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_title('ICEC cross section ' + r'$\text{H}^+ \text{LiH}$')
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', linestyle= '--')
    for r in R:
        label = r'$R=$' + str(round(R*BOHR2ANGSTROM)) + 'A'
        plot_xs(ax, system, r, 0, label)
    #fname = DIR + 'plots/' + system + '.R'+ str(round(R*BOHR2ANGSTROM)) + '.icec.pdf'
    plt.tight_layout()
    fname = DIR + 'plots/' + system + '.R.icec.pdf'
    fig.savefig(fname)
    
def plot_spectrum(system, R, electronE, vi=0, title=None, icec_fixed:ICEC=None, modifier=''):
    fname = DIR + "results/" + system + ".spectrum.E"+ str(round(electronE*HARTREE2EV)) + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.txt"
    results = np.loadtxt(fname, comments='#')
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon_\text{out}$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    
    color = ['tab:red', 'tab:purple', 'tab:blue']
    
    bars = [None] * (v_max+1)
    for vi in range(v_max+1):
        label = r'$v_i=$' + str(vi)
        bars[vi] = ax.bar(results[:,3*vi], results[:,3*vi+1], width=0.002, label=label, color=color[vi])
    labels = [r'$v_i=$' + str(vi) for vi in range(v_max+1)] 
        
    if icec_fixed is not None:
        hbarOmega, energy_out = icec_fixed.energy_relation(electronE)
        xs = icec_fixed.xs(electronE, R)
        bars.append(ax.bar(energy_out*HARTREE2EV, xs*AU2MB, width=0.002, color='black', label='unresolved'))
    labels.append('unresolved')
        
    ax.legend(handles=[bar[0] for bar in bars], labels=labels, ncols=2, fontsize='small', loc='upper right')
    
    fname = DIR + 'plots/' + system + ".spectrum" + modifier + ".E"+ str(round(electronE*HARTREE2EV)) + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.pdf"
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_spectrum_FC(system, R, electronE, vi=0, title=None, icec_fixed:ICEC=None):
    fname = DIR + "results/" + system + ".spectrum-FC.E"+ str(round(electronE*HARTREE2EV)) + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.txt"
    results_FC = np.loadtxt(fname, comments='#')
    
    fname = DIR + "results/" + system + ".spectrum.E"+ str(round(electronE*HARTREE2EV)) + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.txt"
    results_resolved = np.loadtxt(fname, comments='#')
    
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon_\text{out}$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    color_resolved = ['darkred', 'purple', 'darkblue']
    color_FC = ['red', 'violet' ,'lightskyblue']
    for vi in range(v_max+1):
        label = r'$\nu=$' + str(vi)
        ax.bar(results_resolved[:,3*vi], results_resolved[:,3*vi+1], width=0.004, color=color_resolved[vi], label=label)
        ax.bar(results_FC[:,3*vi], results_FC[:,3*vi+1], width=0.002, color=color_FC[vi], label=label + ' FC')
    if icec_fixed is not None:
        hbarOmega, energy_out = icec_fixed.energy_relation(electronE)
        xs = icec_fixed.xs(electronE, R)*AU2MB
        ax.bar(energy_out*HARTREE2EV, xs, width=0.002, color='gray', label='unresolved')
        
    ax.legend(ncols=4, fontsize='small')
    fname = DIR + 'plots/' + system + ".spectrum-FC.E"+ str(round(electronE*HARTREE2EV)) + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.pdf"
    fig.savefig(fname)
    
def plot_morse(icec:IntraICEC, system):
    yshift = (8.066308039 - 7.781734076) * HARTREE2EV 
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,  height_ratios=[0.4, 0.6], figsize=(5,5))
    fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
    ax2.set_xlabel(r'$R$ [$\mathrm{\AA}$]')
    
    ax2.set_ylim(-0.1, 2.5)
    ax1.set_ylim(yshift-0.1, yshift - 0.1 + 0.4/0.6*(2.5+0.1))
    
    r = icec.Morse_D.make_rgrid(rmax=5*ANGSTROM2BOHR)
    V = icec.Morse_D.V(r)
    ax2.plot(r*BOHR2ANGSTROM, V*HARTREE2EV, color='black', label=r'$\mathrm{LiH}$')
    ax2.annotate(r'$\mathrm{LiH}$', (r[-100]*BOHR2ANGSTROM, V[-100]*HARTREE2EV - 0.25))
    
    for vi in range(v_max+1):
        psi = [icec.Morse_D.psi(vi,r_i)/15 + icec.Morse_D.energy(vi)*HARTREE2EV for r_i in r]
        ax2.plot(r*BOHR2ANGSTROM, psi, color='black', lw=1)
    
    V = icec.Morse_Dp.V(r)
    ax1.plot(r*BOHR2ANGSTROM, V*HARTREE2EV + yshift, color='black', ls='--', label=r'$\mathrm{LiH}^+$')
    ax1.annotate(r'$\mathrm{LiH}^+$', (r[-100]*BOHR2ANGSTROM, V[-100]*HARTREE2EV + yshift + 0.05))
    
    psi = [icec.Morse_Dp.psi(0,r_i)/15 + yshift + icec.Morse_Dp.energy(0)*HARTREE2EV for r_i in r]
    ax1.plot(r*BOHR2ANGSTROM, psi, color='black', lw=1)
    
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(bottom=False)
    
    # cut out slanted lines
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    
    fig.text(0.04, 0.5, r'$E$ [eV]', va='center', rotation='vertical')
    
    fname = DIR + 'plots/' + system + ".PES.pdf"
    fig.savefig(fname)

def test_FC_factors(icec_fixed: ICEC, icec:IntraICEC, icec_FC:IntraICEC):
    omega = 10*EV2HARTREE
    print('xs         ', icec.PI_xs_D(0,0,omega))
    print('xs FC      ', icec_FC.PI_xs_D(0,0,omega))
    print('xs FC paper', icec_fixed.PI_xs_B(omega*HARTREE2EV)*MB2AU*0.0153)
    
    for vp in range(0, vp_max+1):
        icec_FC.FC_factor(0,vp)
        
def plot_H_PI_PR(icec:ICEC):
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_title('Hydrogen')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    
    PI_xs = np.array([])
    hbaromega = np.array([])
    for electronE in icec.energyGrid:
        omega = electronE + icec.IP_A
        hbaromega = np.append(hbaromega, [omega*HARTREE2EV])
        xs = icec.PI_xs_A(omega*HARTREE2EV)
        PI_xs = np.append(PI_xs, [xs])
    ax.plot(icec.energyGrid*HARTREE2EV, PI_xs, label = r'$H\to H^+$')
    icec.plot_PR_xs(ax, label = r'$H^+\to H$')
    ax.legend()
    fname = DIR + 'plots/H.PI.PR.pdf'
    fig.savefig(fname)
    

HLi = True
BLi = False
calculation = False
  
if HLi:
    system = 'Hp-LiH'
    title = r'$\text{H}^+ \text{LiH}$'
    
    icec_fixed = ICEC(*input_HLiH_fixed)
    icec_fixed.make_energy_grid(min_kinE*HARTREE2EV, max_kinE*HARTREE2EV, resolution)
    
    icec = IntraICEC(*input_HLiH)
    icec.input_vib_spacing_D(vib_spacing_LiH, vib_spacing_LiHp)
    icec.make_energy_grid(min_kinE, max_kinE, resolution)
    icec.define_Morse_D(*state_LiH, wexe=wexe_LiH)
    icec.define_Morse_Dp(*state_LiHp, wexe=wexe_LiHp)
    icec.define_PI_xs_D(method="resolved")
    
    icec_FC = IntraICEC(*input_HLiH_unresolved)
    icec_FC.define_Morse_D(*state_LiH)
    icec_FC.define_Morse_Dp(*state_LiHp)
    icec_FC.make_energy_grid(min_kinE, max_kinE, resolution)
    icec_FC.define_PI_xs_D(method="FC")
    
    #test_FC_factors(icec_fixed, icec, icec_FC)
    
    plot_morse(icec, 'LiH')
    
    system = 'Hp-LiH'
    header = 'e- + H+ + LiH -> H + LiH+ + e-\n'
    header += f'Number of initial vibrational states: {v_max+1}\n' 
    header += f'Number of final vibrational states: {vp_max+1}\n' 
    
    R=4*ANGSTROM2BOHR
    if calculation:
        calculate_xs_bb(system, header, icec, R, v_max, vp_max)
        calculate_xs_bb(system, header, icec_FC, R, v_max, vp_max, modifier='-FC')
        #calculate_xs_R(system, icec, R, header)
    
    plot_xs_vi(system, icec, R)
    plot_xs_FC(system, icec, R, icec_fixed=icec_fixed)

    if calculation:
        for electronE in electron_energies:
            calculate_spectrum(system, header, icec, R, electronE)
            calculate_spectrum(system, header, icec_FC, R, electronE, modifier='-FC')

    plot_spectrum(system, R, 1*EV2HARTREE, title=title, icec_fixed=icec_fixed)
    plot_spectrum_FC(system, R, 1*EV2HARTREE, title=title, icec_fixed=icec_fixed)

    T = [15, 298, 2000] 
    plot_xs_boltzmann(system, icec, R, T, vib_energies_LiH)
    #xs_vB_vBp(system, icec, 4*ANGSTROM2BOHR)

if BLi:
    system = 'Bp-LiH'
    title = r'$\text{B}^+ \text{LiH}$'
    
    icec = IntraICEC(*input_BLiH)  
    #icec.input_vib_spacing_D(vib_spacing_LiH, vib_spacing_LiHp)
    icec.define_Morse_D(*state_LiH)
    icec.define_Morse_Dp(*state_LiHp)
    icec.make_energy_grid(min_kinE, max_kinE, resolution) 
    
    R = 10 * ANGSTROM2BOHR
    
    header = 'e- + B+ + LiH -> B + LiH+ + e-\n'
    header += f'Number of initial vibrational states: {v_max+1}\n' 
    header += f'Number of final vibrational states: {vp_max+1}\n' 
    header += f'R = {round(R*BOHR2ANGSTROM)} Angstrom'
    header += 'E_in [eV] | xs [Mb]'
        
    #xs_bb(system, header, icec, R, v_max, vp_max)
    
    T = [15, 298, 2000] 
    plot_xs_boltzmann(system, icec, R, T, vib_energies_LiH)
    plot_xs_vi(system, icec, R, title=r"\mathrm{B}^+ + \mathrm{LiH}")

    #for electronE in electron_energies:
    #    calculate_spectrum(system, header, icec, R, electronE)

    #plot_spectrum(system, R, 1*EV2HARTREE, title=title)
    #plot_spectrum(system, R, 5*EV2HARTREE, title=title)