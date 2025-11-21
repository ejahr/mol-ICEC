import numpy as np
import matplotlib.pyplot as plt
from input.HLiH import LiH, LiHp, Hp_LiH, Bp_LiH
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.morse import Morse
from icec.constants import Units
from plotting.pes import plot_PES, plot_diss_at_L
from plotting.spectrum import plot_spectrum, plot_spectrum_bc, plot_spectrum_FC
from plotting.cross_sections import plot_xs_vi, plot_xs_FC, plot_xs_boltzmann, plot_xs_R, plot_xs_vB_vBp

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 16})
width, height = 6, 4

DIR = '/home/elena/intraICEC/dimers/'

def calculate_xs_bb(system, header, icec: IntraICEC, R, vD_max=None, vDp_max=None, modifier=''):
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    if vDp_max is None:
        vDp_max = icec.Morse_Dp.vmax
    header += f'Number of initial vibrational states: {vD_max+1}/{icec.Morse_D.vmax+1}\n'
    header += f'Number of final vibrational states: {vDp_max+1}\n'  
    header += 'E_in [eV] | xs [Mb]'
    xs_array = icec.energyGrid*Units.HARTREE2EV
    for v in range(vD_max+1):
        xs = icec.xs_vD(R, v, vDp_max)
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR + f"results/{system}.xs{modifier}.R"+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)
    
def calculate_xs_R(system, header, icec, R, vD_max=None, vDp_max=None):
    for r in R:
        headerR = header + f'R_AD = {round(r*Units.BOHR2ANGSTROM)} Angstrom\n'
        calculate_xs_bb(system, headerR, icec, r, vD_max, vDp_max)
    
def calculate_xs_bc(system, header, icec: IntraICEC, R, vD_max=None, modifier=''):
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    header += f'Number of initial vibrational states: {vD_max+1}/{icec.Morse_D.vmax+1}\n'
    header += f'Dissociative states of D^+: \
        Max energy = {round(icec.Morse_Dp.diss_energies[-1]*Units.HARTREE2EV)}, \
        Box length = {round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM)} Angstrom\n' 
    header += 'E_in [eV] | xs [Mb]'
    xs_array = icec.energyGrid*Units.HARTREE2EV
    for vD in range(vD_max+1):
        xs = icec.xs_vD_continuum(R, vD)*Units.AU2MB
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR + f"results/{system}.xs{modifier}.bc.R{str(round(R*Units.BOHR2ANGSTROM))}.L{str(round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM))}.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)    
        
def calculate_xs_bc_R(system, icec, R, header):
    for r in R:
        headerR = header + f'R_AD = {round(r*Units.BOHR2ANGSTROM)} Angstrom\n'
        calculate_xs_bc(system, headerR, icec, r, LiH.v_max)

def calculate_spectrum(system, header, icec: IntraICEC, R, electronE, vD_max=None, vDp_max=None, modifier=''): 
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    if vDp_max is None:
        vDp_max = icec.Morse_Dp.vmax
    header += f'Number of initial vibrational states: {vD_max+1}/{icec.Morse_D.vmax+1}\n'
    header += f'Number of final vibrational states: {vDp_max+1}\n'
    header += "E_in = " + str(round(electronE*Units.HARTREE2EV)) + " eV\n"
    header += "| E_out [eV] : xs [Mb] |"  
    spectrum_all_vi = np.array([]) 
    for vi in range(vD_max+1):
        spectrum = icec.spectrum(electronE, R, vi, vDp_max)
        if spectrum_all_vi.size == 0:
            spectrum_all_vi = spectrum
        else:
            spectrum_all_vi = np.hstack((spectrum_all_vi, spectrum))          
    fname = DIR + f"results/{system}.spectrum{modifier}.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    np.savetxt(fname, spectrum_all_vi, fmt='%1.3e', header=header)  
    
def calculate_spectrum_bc(system, header, icec: IntraICEC, R, electronE, modifier=''): 
    header += "Spectrum for ICEC with E_in = " + str(round(electronE*Units.HARTREE2EV)) + " eV\n"
    header += f'Box length for dissociative states of D^+: {round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM)} Angstrom\n' 
    header += "E_out [eV] | xs [Mb] | diss_energy [eV]"  
    vD = 0
    spectrum = icec.spectrum_bc(electronE, R, vD)   
    fname = DIR + f'results/{system}.spectrum{modifier}.bc.v0.E{str(round(electronE*Units.HARTREE2EV))}.R{str(round(R*Units.BOHR2ANGSTROM))}.L{str(round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM))}.txt'
    np.savetxt(fname, spectrum, fmt='%1.3e', header=header)  
 

def test_FC_factors(icec_fixed: ICEC, icec:IntraICEC, icec_FC:IntraICEC):
    omega = 10*Units.EV2HARTREE
    print('xs         ', icec.PI_xs_D(0,0,omega))
    print('xs FC      ', icec_FC.PI_xs_D(0,0,omega))
    print('xs FC paper', icec_fixed.PI_xs_B(omega*Units.HARTREE2EV)*Units.MB2AU*0.0153)             
    
    for vp in range(0, LiH.v_max+1):
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
        hbaromega = np.append(hbaromega, [omega*Units.HARTREE2EV])
        xs = icec.PI_xs_A(omega*Units.HARTREE2EV)
        PI_xs = np.append(PI_xs, [xs])
    ax.plot(icec.energyGrid*Units.HARTREE2EV, PI_xs, label = r'$H\to H^+$')
    icec.plot_PR_xs(ax, label = r'$H^+\to H$')
    ax.legend()
    fname = DIR + 'plots/H.PI.PR.pdf'
    fig.savefig(fname)
    
def calculate_roots(Morse:Morse, fname, max_energy:float=1*Units.EV2HARTREE, num:int=500):
    roots, root_estimates = Morse.save_diss_states(fname, max_energy, num)
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_title('Dissociative states')
    ax.set_xlabel(r'$E$ [eV]')
    ax.set_ylabel(r'$E$ [a.u.]')
    ax.set_yscale('log')
    ax.bar(root_estimates*Units.HARTREE2EV, root_estimates/2, width=0.005, color='tab:blue', label='estimates')
    ax.bar(roots*Units.HARTREE2EV, roots, width=0.005, color='tab:red', label='roots')
    ax.legend()
    plt.tight_layout()
    fname = DIR + 'plots/LiHp.roots.' + str(round(Morse.box_length*Units.BOHR2ANGSTROM)) + 'A.pdf'
    fig.savefig(fname)

HLi = True
BLi = False
calculation_bb = 0
calculation_bc = 1
plot_bb = 0
plot_bc = 1

#https://doi.org/10.1021/jp9921295
R = 2 * Units.ANGSTROM2BOHR

electronE = 1*Units.EV2HARTREE
R = 6*Units.ANGSTROM2BOHR
L = 8*Units.ANGSTROM2BOHR
#L = 16*Units.ANGSTROM2BOHR
R_list = np.array([6,8,10]) * Units.ANGSTROM2BOHR

min_kinE = 0.01 * Units.EV2HARTREE
max_kinE = 9 * Units.EV2HARTREE
max_dissE = 1.3 * Units.EV2HARTREE
resolution = 1000

if HLi:
    system = 'Hp-LiH'
    title = r'$\text{H}^+ \text{LiH}$'
    
    icec_fixed = ICEC(*Hp_LiH.input_fixed)
    icec_fixed.make_energy_grid(min_kinE*Units.HARTREE2EV, LiH.max_kinE_unresolved*Units.HARTREE2EV, resolution)
    
    icec = IntraICEC(*Hp_LiH.input)
    icec.input_vib_spacing_D(LiH.vib_spacing, LiHp.vib_spacing)
    icec.make_energy_grid(min_kinE, max_kinE, resolution)
    icec.define_Morse_D(*LiH.morse_parameters, wexe=LiH.wexe)
    icec.define_Morse_Dp(*LiHp.morse_parameters, wexe=LiHp.wexe)
    icec.define_PI_xs_D(method="resolved")
    
    icec_FC = IntraICEC(*Hp_LiH.input_unresolved)
    icec_FC.define_Morse_D(*LiH.morse_parameters, wexe=LiH.wexe)
    icec_FC.define_Morse_Dp(*LiHp.morse_parameters, wexe=LiHp.wexe)
    icec_FC.make_energy_grid(min_kinE, max_kinE, resolution)
    icec_FC.define_PI_xs_D(method="FC")
    
    icec_FC.Morse_Dp.define_box(L)
    fname = DIR + 'data/LiH/LiHp.diss_energies.L' + str(round(L*Units.BOHR2ANGSTROM)) + 'A.txt'
    calculate_roots(icec_FC.Morse_Dp, fname, max_energy=1*Units.EV2HARTREE, num=1000)
    icec_FC.Morse_Dp.load_diss_states(fname)
    
    #plot_H_PI_PR(icec_fixed)
    #test_FC_factors(icec_fixed, icec, icec_FC)
    energy_difference = (7.974721285 - 7.776735464) * Units.HARTREE2EV # difference at R=inf
    plot_PES(icec_FC, 'LiH', L, energy_difference)
    plot_diss_at_L(icec_FC.Morse_Dp, system, L)
    
    system = 'Hp-LiH'
    header = 'e- + H+ + LiH -> H + LiH+ + e-\n'

    if calculation_bb:
        calculate_xs_bb(system, header, icec, R, LiH.v_max, LiHp.v_max)
        calculate_xs_bb(system, header, icec_FC, R, modifier='-FC')
        calculate_xs_R(system, header, icec, R_list, LiH.v_max, LiHp.v_max)        
        calculate_spectrum(system, header, icec, R, electronE, LiH.v_max, LiHp.v_max)
        calculate_spectrum(system, header, icec_FC, R, electronE, LiH.v_max, modifier='-FC')
        
    if calculation_bc:
        calculate_xs_bc(system, header, icec_FC, R, vD_max=5, modifier='-FC')
        calculate_spectrum_bc(system, header, icec_FC, R, electronE, modifier='-FC')
    
    if plot_bb:
        plot_xs_vi(system, icec, R, LiH.v_max, icec_fixed)
        plot_xs_R(system, icec, R_list, icec_fixed=icec_fixed)
        plot_spectrum(system, icec, R, 1*Units.EV2HARTREE, LiH.v_max, icec_fixed=icec_fixed)
        plot_spectrum_FC(system, R, 1*Units.EV2HARTREE, LiH.v_max)
    
        T = [15, 298, 2000] 
        plot_xs_boltzmann(system, icec, R, T, LiH.v_max, LiH.vib_energies)
        plot_xs_vB_vBp(system, icec, R, LiH.v_max, LiHp.v_max)
        
    if plot_bc:
        plot_xs_FC(system, icec_FC, R, icec_fixed)
        plot_spectrum_bc(system, icec_FC, R, electronE, vi=0)
        

if BLi:
    system = 'Bp-LiH'
    title = r'$\text{B}^+ \text{LiH}$'
    
    icec = IntraICEC(*Bp_LiH.input)  
    #icec.input_vib_spacing_D(vib_spacing_LiH, vib_spacing_LiHp)
    icec.define_Morse_D(*LiH.morse_parameters)
    icec.define_Morse_Dp(*LiHp.morse_parameters)
    icec.make_energy_grid(min_kinE, max_kinE, resolution) 
    
    R = 10 * Units.ANGSTROM2BOHR
    
    header = 'e- + B+ + LiH -> B + LiH+ + e-\n'
    header += f'Number of initial vibrational states: {LiH.v_max+1}\n' 
    header += f'Number of final vibrational states: {LiHp.v_max+1}\n' 
    header += f'R = {round(R*Units.BOHR2ANGSTROM)} Angstrom'
    header += 'E_in [eV] | xs [Mb]'
        
    #xs_bb(system, header, icec, R, v_max, vp_max)
    
    T = [15, 298, 2000] 
    plot_xs_boltzmann(system, icec, R, T, LiH.vib_energies)
    plot_xs_vi(system, icec, R, title=r"\mathrm{B}^+ + \mathrm{LiH}")

    #for electronE in electron_energies:
    #    calculate_spectrum(system, header, icec, R, electronE)

    #plot_spectrum(system, R, 1*Units.EV2HARTREE, title=title)
    #plot_spectrum(system, R, 5*Units.EV2HARTREE, title=title)