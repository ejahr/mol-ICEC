import sys
import numpy as np
from HLiH_input import * # defines constants and parameters
#sys.path.insert(0, '/mnt/home/elena/icec_project') % on linux servers
sys.path.insert(0, '/home/elena/icec-project')
from crosssection.icec.intraIcec import IntraICEC
from crosssection.icec.constants import *

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 14})

DIR = '/home/elena/icec-project/dimers/'

#https://doi.org/10.1021/jp9921295
R = 2 * ANGSTROM2BOHR
#R = 10 * ANGSTROM2BOHR
R = np.array([2,4,6,8,10]) * ANGSTROM2BOHR

min_kinE = 0.01 * EV2HARTREE
max_kinE = 10 * EV2HARTREE
resolution = 200

v_max = 2
vp_max = 5

electron_energies = np.array([1, 5]) * EV2HARTREE

def xs_vB_vBp(system, icec: IntraICEC, R):
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
                xs = icec.xs_vB_vBp(R, vi, vf)
                ax.plot(energies, xs, label=label)
            ax.legend()
            pdf.savefig(fig)  #, bbox_inches = "tight"
            plt.close(fig) 
    

def xs_bb(system, header, icec: IntraICEC, R, v_max, vp_max):
    xs_array = icec.energyGrid*HARTREE2EV
    for v in range(v_max+1):
        xs = icec.xs_vB(R, v, vp_max)
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR + "results/" + system + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)
    
def calculate_xs_R(system, icec, R, header):
    for r in R:
        headerR = header + f'R = {round(r*BOHR2ANGSTROM)} Angstrom'
        headerR += 'E_in [eV] | xs [Mb]'
        xs_bb(system, headerR, icec, r, v_max, vp_max)

def calculate_spectrum(system, header, icec: IntraICEC, R, electronE): 
    new_header = header + "E_in = " + str(round(electronE*HARTREE2EV)) + " eV\n"
    new_header += "| E_out [eV] : xs [Mb] |"  
    spectrum_all_vi = np.array([]) 
    for vi in range(v_max+1):
        spectrum = icec.spectrum(electronE, R, vi, vp_max)
        if spectrum_all_vi.size == 0:
            spectrum_all_vi = spectrum
        else:
            spectrum_all_vi = np.hstack((spectrum_all_vi, spectrum))          
    fname = DIR + "results/" + system + ".spectrum."+ str(round(electronE*HARTREE2EV)) + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.txt"
    np.savetxt(fname, spectrum_all_vi, fmt='%1.3e', header=new_header)  
    
def read_results_file(system, R):
    file_path = DIR + "results/" + system + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.txt"
    results = np.loadtxt(file_path, comments='#')
    return results

def plot_xs(ax, system, R, v_B, label, **kwargs):
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    # ax.plot(icec.energyGrid * HARTREE2EV,  icec.PI_xs_B(v_B, 0, icec.energyGrid + icec.IP_A)*AU2MB, label=r'$\sigma_\text{PI}$')
    results = read_results_file(system, R)
    ax.plot(results[:,0], results[:, v_B+1], label=label, **kwargs)
    ax.legend()
    
def plot_xs_vi(system, icec: IntraICEC, R):
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_title('ICEC cross section ' + r'$\text{H}^+ \text{LiH}$')
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', linestyle= '--')
    for vi in range(0, v_max+1):
        label = r'$v_{LiH}=$' + str(vi)
        plot_xs(ax, system, R, vi, label)
    fname = DIR + 'plots/' + system + '.vB.R'+ str(round(R*BOHR2ANGSTROM)) + 'icec.pdf'
    fig.savefig(fname)
    
def plot_xs_boltzmann(system, icec: IntraICEC, R, T, vib_energies):
    vib_energies_LiH
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_title('ICEC cross section ' + r'$\text{H}^+ \text{LiH}$')
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.set_yscale('log')
    ax.set_xlim(0,5)
    ax.set_ylim(1e-3, 1e2)
    ax.grid(True)
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', linestyle= '--')
    results = read_results_file(system, R)
    for t in T:
        norm = sum(np.exp(-vib_energies[vi]/KB/t) for vi in range(v_max+1))
        avg = 0
        label = r'$T=$' + str(t) + 'K'
        for vi in range(0, v_max+1):
            avg += np.exp(-vib_energies[vi]/KB/t) * results[:, vi+1]
        ax.plot(results[:,0], avg/norm, label=label)
    for vi in range(v_max + 1):
        plot_xs(ax, system, R, vi, r'$v_{LiH}=$'+str(vi), linestyle=':')
    ax.legend()
    plt.tight_layout()
    fname = DIR + 'plots/' + system + '.boltzmann.R'+ str(round(R*BOHR2ANGSTROM)) + 'icec.pdf'
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
    
def plot_spectrum(system, R, electronE, vi=0):
    fname = DIR + "results/" + system + ".spectrum."+ str(round(electronE*HARTREE2EV)) + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.txt"
    results = np.loadtxt(fname, comments='#')
    
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_title('ICEC cross section ' + r'$\text{H}^+ \text{LiH}$')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon_\text{out}$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    for vi in range(v_max+1):
        label = r'$\epsilon=$' + str(round(electronE*HARTREE2EV)) + r', $v_{LiH}=$' + str(vi)
        ax.bar(results[:,3*vi], results[:,3*vi+1],width=0.002, label=label)
    ax.legend()
    fname = DIR + 'plots/' + system + ".spectrum.E"+ str(round(electronE*HARTREE2EV)) + '.R'+ str(round(R*BOHR2ANGSTROM)) + ".icec.pdf"
    fig.savefig(fname)
    

  
icec = IntraICEC(*input_HLiH)
icec.input_vib_spacing_B(vib_spacing_LiH, vib_spacing_LiHp)
icec.make_energy_grid(min_kinE, max_kinE, resolution)
  
system = 'Hp-LiH'
header = 'e- + H+ + LiH -> H + LiH+ + e-\n'
header += f'Number of initial vibrational states: {v_max+1}\n' 
header += f'Number of final vibrational states: {vp_max+1}\n' 

#calculate_xs_R(system, icec, R, header)
#plot_xs_vi(system, icec, R=4*ANGSTROM2BOHR)

for electronE in electron_energies:
    r = 4 * ANGSTROM2BOHR
    #calculate_spectrum(system, header, icec, r, electronE)

#plot_spectrum(system, 4*ANGSTROM2BOHR, 1*EV2HARTREE)

T = [15, 298, 2000] 
plot_xs_boltzmann(system, icec, 4*ANGSTROM2BOHR, T, vib_energies_LiH)
xs_vB_vBp(system, icec, 4*ANGSTROM2BOHR)

#icec = IntraICEC(*input_BLiH)  
#icec.input_vib_spacing_B(vib_spacing_LiH, vib_spacing_LiHp)
#icec.make_energy_grid(min_kinE, max_kinE, resolution)  
  
#system = 'Bp-LiH'
#header = 'e- + B+ + LiH -> B + LiH+ + e-\n'
#header += f'Number of initial vibrational states: {v_max+1}\n' 
#header += f'Number of final vibrational states: {vp_max+1}\n' 
#header += f'R = {round(R*BOHR2ANGSTROM)} Angstrom'
#header += 'E_in [eV] | xs [Mb]'
    
#xs_bb(system, header, icec, R, v_max, vp_max)
#plot_xs(system, icec, R, 0)
