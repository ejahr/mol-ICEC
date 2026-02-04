import numpy as np
import matplotlib.pyplot as plt
from config import DIR
from input.HLiH import LiH, LiHp, Hp_LiH
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.morse import Morse
from icec.constants import Units, Constants
from plot.pes import plot_PES, plot_diss_at_L
from plot.spectrum import plot_spectrum, plot_spectrum_bc, plot_spectrum_FC
from plot.cross_sections import plot_xs_vi, plot_xs_FC_bb, plot_xs_FC, plot_xs_boltzmann, plot_xs_boltzmann_FC

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 16})
width, height = 6, 4

def extend_header(header, icec:IntraICEC, R=None, vD_max=None, vDp_max=None, electronE=None, result_type=None):
    if result_type == 'spectrum':
        header += "ICEC electron spectrum at E_in = " + str(round(electronE*Units.HARTREE2EV)) + " eV\n"
    if result_type == 'xs':
        header += "ICEC cross section\n"
    header += f'R = {str(round(R*Units.BOHR2ANGSTROM))} Angstrom\n' 
    if vD_max is not None:
        header += f'Number of initial vibrational states: {vD_max+1}/{icec.Morse_D.vmax+1}\n'
    if vDp_max is not None:
        header += f'Number of final vibrational states: {vDp_max+1}\n'
    else:
        header += f'Dissociative states of D^+: \
            Max energy = {round(icec.Morse_Dp.diss_energies[-1]*Units.HARTREE2EV)} eV, \
            Box length = {round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM)} Angstrom\n' 
    if result_type == 'xs':
        header += 'E_in [eV] | xs [Mb]' 
    return header

# ========= Running calculations and saving results ============

def calculate_xs_bb(system, header, icec: IntraICEC, R, vD_max=None, vDp_max=None, modifier=''):
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    if vDp_max is None:
        vDp_max = icec.Morse_Dp.vmax
    header = extend_header(header, icec, R=R, vD_max=vD_max, vDp_max=vDp_max, result_type='xs')
    xs_array = icec.energyGrid*Units.HARTREE2EV
    for v in range(vD_max+1):
        xs = icec.xs_vD(R, v, vDp_max)
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR + f"results/{system}.xs{modifier}.R"+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)
    
def calculate_xs_R(system, header, icec, R, vD_max=None, vDp_max=None):
    for r in R:
        calculate_xs_bb(system, header, icec, r, vD_max, vDp_max)
    
def calculate_xs_bc(system, header, icec: IntraICEC, R, vD_max=None, modifier=''):
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    header = extend_header(header, icec, R, vD_max)
    xs_array = icec.energyGrid*Units.HARTREE2EV
    for vD in range(vD_max+1):
        xs = icec.xs_vD_continuum(R, vD)*Units.AU2MB
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR + f"results/{system}.xs{modifier}.bc.R{str(round(R*Units.BOHR2ANGSTROM))}.L{str(round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM))}.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)    
        
def calculate_xs_bc_R(system, icec, R, header):
    for r in R:
        calculate_xs_bc(system, header, icec, r, LiH.v_max)

def calculate_spectrum(system, header, icec_el:ICEC, icec: IntraICEC, R, electronE, vD_max=None, vDp_max=None, modifier=''): 
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    if vDp_max is None:
        vDp_max = icec.Morse_Dp.vmax
    header = extend_header(header, vD_max, vDp_max, result_type='spectrum')
    header += f"Electronic result: E_out = {icec_el.electronE_f(electronE)*Units.HARTREE2EV} eV, xs = {icec_el.xs(electronE, R)*Units.AU2MB} Mb"
    header += "| E_out [eV], xs [Mb], v_Dp |"  
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
    header = extend_header(icec, R, electronE=electronE, result_type='spectrum')
    header += "E_out [eV] | xs [Mb] | diss_energy [eV]"  
    vD = 0
    spectrum = icec.spectrum_bc(electronE, R, vD)   
    fname = DIR + f'results/{system}.spectrum{modifier}.bc.v0.E{str(round(electronE*Units.HARTREE2EV))}.R{str(round(R*Units.BOHR2ANGSTROM))}.L{str(round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM))}.txt'
    np.savetxt(fname, spectrum, fmt='%1.3e', header=header)  
 
# ============= Information ===============

def test_FC_factors(icec_el: ICEC, icec:IntraICEC, icec_FC:IntraICEC, R:float):
    electronE = 1*Units.EV2HARTREE
    omega = electronE + icec_el.IP_A
    
    print('\n--- Test FC approx ---')
    for vf in range(5):
        print(f'0->{vf}/elec')
        print(' PI     ', icec.PI_xs_D(0,vf,omega)/icec_el.PI_xs_B(omega*Units.HARTREE2EV)/Units.MB2AU)
        print(' ICEC   ', icec.xs(electronE,R,0,vf)/icec_el.xs(electronE,R))
        print(' FC ICEC', icec_FC.xs(electronE,R,0,vf)/icec_el.xs(electronE,R))     
        print(' FC     ', icec_FC.FC_factor(0,vf))
        
    print('\n--- v-ratios ---')
    for vf in range(1,5):
        print(f'0->{vf}/0->{vf-1}')
        print(' PI     ', icec.PI_xs_D(0,vf,omega)/icec.PI_xs_D(0,vf-1,omega))
        print(' ICEC   ', icec.xs(electronE,R,0,vf)/icec.xs(electronE,R,0,vf-1))
        print(' FC ICEC', icec_FC.xs(electronE,R,0,vf)/icec_FC.xs(electronE,R,0,vf-1))
        print(' FC     ', icec_FC.FC_factor(0,vf)/icec_FC.FC_factor(0,vf-1))
    
def print_boltzmann_probabilities(icec:IntraICEC, T):
    print('\n--- Boltzmann probabilities ---')
    vmax = icec.Morse_D.vmax
    vmax = 5
    for t in T:
        print(f'T = {t} K')
        # add De to energy(vi) to get positive values which increases numerical stability
        norm = sum(np.exp(-(icec.Morse_D.energy(vi)+icec.Morse_D.De)/Constants.KB/t) 
                        for vi in range(vmax+1)
                        )
        for vi in range(vmax):
            value = np.exp(-(icec.Morse_D.energy(vi)+icec.Morse_D.De)/Constants.KB/t) / norm
            print(f'vi = {vi}: {value}')  
            
            
def print_PI_crosssection(icec:IntraICEC):
    omega = 14.6*Units.EV2HARTREE
    print("\n--- PI cross section at omega = 14.6 eV ---")
    print(f' H    : {icec.PI_xs_A(omega)*Units.AU2MB} Mb')
    print(f' omega: {omega*Units.HARTREE2EV} eV')
    print(f' LiH  : {icec.PI_xs_D_electronic(omega)*Units.AU2MB} Mb')
        
        
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
    
# ========= Pre calculate roots of box for dissociative states =============
    
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


# ================ Main ===================

# ===== Define which parts are active =====
calc_roots      = 0
bb              = 1
bc              = 0
FC              = 0
plot            = 0
calculate       = 1
spectra         = 0
cross_section   = 1
temp_dependence = 0
plot_info       = 0
print_info      = 0

electronE   = 1*Units.EV2HARTREE
R           = 6*Units.ANGSTROM2BOHR
L           = 8*Units.ANGSTROM2BOHR
#L = 16*Units.ANGSTROM2BOHR
R_list      = np.array([6,8,10]) * Units.ANGSTROM2BOHR
T           = [15, 300, 1500] 

vD_max_bc   = 5
min_kinE    = 0.01 * Units.EV2HARTREE
max_kinE    = 9 * Units.EV2HARTREE
max_dissE   = 1.3 * Units.EV2HARTREE
resolution  = 1000

system = 'Hp-LiH'
title = r'$\text{H}^+ \text{LiH}$'

# --------- Define ICEC classes for calculating ICEC cross sections ---------
# --- ICEC with vibrationally resolved photoionization cross section of D ---
icec = IntraICEC(*Hp_LiH.input)
icec.input_vib_spacing_D(LiH.vib_spacing, LiHp.vib_spacing)
icec.make_energy_grid(min_kinE, max_kinE, resolution)
icec.define_Morse_D(*LiH.morse_parameters, wexe=LiH.wexe)
icec.define_Morse_Dp(*LiHp.morse_parameters, wexe=LiHp.wexe)
IP_adiabatic = icec.IP_D - (icec.Morse_D.energy(0)+ icec.Morse_D.De) + (icec.Morse_Dp.energy(0)+ icec.Morse_Dp.De)
icec.IP_D = IP_adiabatic
icec.define_PI_xs_D(method="resolved")

# --- ICEC within Franck-Condon approximation for photoionization of D ---
icec_FC = IntraICEC(*Hp_LiH.input_unresolved)
icec_FC.IP_D = IP_adiabatic
icec_FC.define_Morse_D(*LiH.morse_parameters, wexe=LiH.wexe)
icec_FC.define_Morse_Dp(*LiHp.morse_parameters, wexe=LiHp.wexe)
icec_FC.make_energy_grid(min_kinE, max_kinE, resolution)
icec_FC.define_PI_xs_D(method="FC")

icec_FC.Morse_Dp.define_box(L)
fname = DIR + 'data/LiH/LiHp.diss_energies.L' + str(round(L*Units.BOHR2ANGSTROM)) + 'A.txt'
if calc_roots:
    calculate_roots(icec_FC.Morse_Dp, fname, max_energy=max_dissE, num=1000)
icec_FC.Morse_Dp.load_diss_states(fname)

# --- electronic ICEC without any nuclear dynamics ---
icec_el = ICEC(*Hp_LiH.input_electronic)
IP_vertical = icec_el.IP_B + (icec.Morse_Dp.V(icec.Morse_D.re) + icec.Morse_Dp.De)
icec_el.IP_B = IP_vertical
icec_el.make_energy_grid(min_kinE*Units.HARTREE2EV, LiH.max_kinE_unresolved*Units.HARTREE2EV, resolution)


if print_info:
    print('\n===== Info =====')
    print(f"vertical ionization energy {IP_vertical*Units.HARTREE2EV} eV")
    print(f"adiabatic ionizaton energy {IP_adiabatic*Units.HARTREE2EV} eV")
    energy_diff_at_inf = (7.974721285 - 7.776735464) * Units.HARTREE2EV # energy difference at R=inf
    print(f"energy diff at R=inf       {energy_diff_at_inf} eV")
    test_FC_factors(icec_el, icec, icec_FC, R)
    print_boltzmann_probabilities(icec_FC, T)
    print_PI_crosssection(icec_FC)
    

system = 'Hp-LiH'
header = 'e- + H+ + LiH -> H + LiH+ + e-\n'

if calculate:
    if bb:
        if cross_section:
            calculate_xs_bb(system, header, icec, R, LiH.v_max, LiHp.v_max)
            if FC:
                calculate_xs_bb(system, header, icec_FC, R, modifier='-FC')
            #calculate_xs_R(system, header, icec, R_list, LiH.v_max, LiHp.v_max)  
        if spectra:      
            calculate_spectrum(system, header, icec_el, icec, R, electronE, LiH.v_max, LiHp.v_max)
            if FC:
                calculate_spectrum(system, header, icec_el, icec_FC, R, electronE, LiH.v_max, modifier='-FC')

    if bc:
        if cross_section:
            calculate_xs_bc(system, header, icec_FC, R, vD_max_bc, modifier='-FC')
        if spectra:
            calculate_spectrum_bc(system, header, icec_FC, R, electronE, modifier='-FC')

if plot:
    if bb:
        if cross_section:
            plot_xs_FC_bb(system, icec_FC, R, icec_el)
        if spectra:
            plot_spectrum(system, icec, R, 1*Units.EV2HARTREE, LiH.v_max, icec_el=icec_el)
            if FC:
                plot_spectrum_FC(system, R, 1*Units.EV2HARTREE, LiH.v_max, icec_el=icec_el)
        #plot_xs_boltzmann(system, icec, R, T, LiH.v_max, LiH.vib_energies)
    
    if bc:
        if cross_section:
            plot_xs_FC(system, icec_FC, R, icec_el)
        if spectra:   
            plot_spectrum_bc(system, icec_FC, R, electronE, vi=0, icec_el=icec_el)
        if temp_dependence:
            plot_xs_boltzmann_FC(system, icec_FC, R, T, vD_max_bc, icec_el=icec_el)

if plot_info:
    plot_diss_at_L(icec_FC.Morse_Dp, "LiH", L)
    plot_PES(icec_FC, 'LiH', L, energy_diff_at_inf)
    #plot_H_PI_PR(icec_el)