import numpy as np
import matplotlib.pyplot as plt
from config import DIR
from input.HLiH import LiH, LiHp, Hp_LiH
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.morse import Morse
from icec.constants import Units, Constants
from plot import cross_sections, pes, spectrum
import plot.config

def extend_header(header:str, icec:IntraICEC, R:float=None, vD_max:int=None, vDp_max:int=None, electronE:float=None, result_type:str=None):
    if result_type == 'spectrum':
        header += f"ICEC electron spectrum at E_in = {round(electronE*Units.HARTREE2EV)} eV\n"
    if result_type == 'xs':
        header += "ICEC cross section\n"
    header += f'R = {round(R*Units.BOHR2ANGSTROM,3)} Angstrom\n' 
    if vD_max is not None:
        header += f'Number of initial vibrational states: {vD_max+1}\n'
    if vDp_max is not None:
        header += f'Number of final vibrational states: {vDp_max+1}\n'
    else:
        header += "Dissociative states of D+: " + \
        f"Max energy = {round(icec.Morse_Dp.diss_energies[-1]*Units.HARTREE2EV,1)} eV, " + \
        f"Box length = {round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM)} Angstrom\n" 
    if result_type == 'xs':
        header += 'E_in [eV] | xs [Mb]' 
    return header

# ========= Running calculations and saving results ============

def calculate_xs_bb(system, header, icec: IntraICEC, R, vD_max=None, vDp_max=None, modifier=''):
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    if vDp_max is None:
        vDp_max = icec.Morse_Dp.vmax
    header = extend_header(header, icec, R, vD_max, vDp_max, result_type='xs')
    xs_array = icec.energyGrid*Units.HARTREE2EV
    for v in range(vD_max+1):
        xs = icec.xs_vD(R, v, vDp_max)
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR + f"results/{system}.xs{modifier}.R{round(R*Units.BOHR2ANGSTROM)}.txt"
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
    file_path = DIR + f"results/{system}.xs{modifier}.bc.R{round(R*Units.BOHR2ANGSTROM)}.L{round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM)}.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)    
        
def calculate_xs_bc_R(system, icec, R, header):
    for r in R:
        calculate_xs_bc(system, header, icec, r, LiH.v_max)

def calculate_spectrum(system: str, header:str, icec_el:ICEC, icec: IntraICEC, R:float, electronE:float, vD_max:int=None, vDp_max:int=None, modifier:str=''): 
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    if vDp_max is None:
        vDp_max = icec.Morse_Dp.vmax
    header = extend_header(header, icec, R, vD_max, vDp_max, electronE, result_type='spectrum')
    header += f"Electronic result: E_out = {icec_el.electronE_f(electronE)*Units.HARTREE2EV} eV, xs = {icec_el.xs(electronE, R)*Units.AU2MB} Mb\n"
    header += "| v_D, v_Dp, E_out [eV], xs [Mb] |"  
    spectrum_all_vD = np.array([]) 
    for vD in range(vD_max+1):
        spectrum = icec.spectrum(electronE, R, vD, vDp_max)
        if spectrum_all_vD.size == 0:
            spectrum_all_vD = spectrum
        else:
            spectrum_all_vD = np.hstack((spectrum_all_vD, spectrum))          
    fname = DIR + f"results/{system}.spectrum{modifier}.E{round(electronE*Units.HARTREE2EV)}.R{round(R*Units.BOHR2ANGSTROM)}.txt"
    np.savetxt(fname, spectrum_all_vD, fmt='%1.3e', header=header)  
    
def calculate_spectrum_bc(system: str, header: str, icec: IntraICEC, R: float, electronE: float, vD_max:int=None, modifier:str=''): 
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    header = extend_header(header, icec, R, vD_max, electronE=electronE, result_type='spectrum')
    header += "| vD, diss_energy [eV], E_out [eV], xs [Mb] |"  
    spectrum_all_vD = icec.spectrum_bc(electronE, R, vD=0)  
    for vD in range(1, vD_max+1):
        spectrum = icec.spectrum_bc(electronE, R, vD) 
        spectrum_all_vD = np.hstack((spectrum_all_vD, spectrum))  
    fname = DIR + f'results/{system}.spectrum{modifier}.bc.E{round(electronE*Units.HARTREE2EV)}.R{round(R*Units.BOHR2ANGSTROM)}.L{round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM)}.txt'
    np.savetxt(fname, spectrum_all_vD, fmt='%1.3e', header=header)  
 
# ============= Information ===============

def test_FC_factors(icec_el: ICEC, icec:IntraICEC, icec_FC:IntraICEC, R:float):
    electronE = 4*Units.EV2HARTREE
    omega = electronE + icec_el.IP_A
    vi = 0
    
    # https://doi.org/10.1063/1.479970
    FC_abinitio = [[0.0153, 0.0292, 0.0305, 0.0214, 0.0103, 0.0031, 0.0004],
                   [0.0610, 0.0823, 0.0643, 0.0366, 0.0157, 0.0045, 0.0006],
                   [0.1252, 0.1033, 0.0497, 0.0188, 0.0062, 0.0016, 0.0002],
                   [0.1749, 0.0665, 0.0097, 0.0003, 0.0000, 0.0000, 0.0000],
                   [0.1866, 0.0146, 0.0036, 0.0105, 0.0068, 0.0022, 0.0003],
                   [0.1619, 0.0021, 0.0349, 0.0276, 0.0121, 0.0035, 0.0005]]
    
    print('\n--- FC factor ---')
    print('tot b-b from 0')
    FC_abinitio_sum = sum(FC_abinitio[vi][vf] for vf in range(7))
    FC_Morse_sum = sum(icec_FC.FC_factor(vi,vf) for vf in range(icec_FC.Morse_Dp.vmax+1))
    print(' FC ab initio', FC_abinitio_sum)
    print(' FC Morse    ', FC_Morse_sum)
    print(' ratio       ', FC_Morse_sum/FC_abinitio_sum)
    
    for vf in range(5):
        print(f'{vi}->{vf}')
        print(' PI / PI elec', icec.PI_xs_D(vi,vf,omega)/icec_el.PI_xs_B(omega*Units.HARTREE2EV)/Units.MB2AU)
        print(' FC Morse    ', icec_FC.FC_factor(vi,vf))
        print(' FC ab initio', FC_abinitio[vi][vf])
        
    print(f'\n--- ICEC Cross section [Mb] at {round(electronE*Units.HARTREE2EV,1)} eV---')
    xs_tot_FC_abinitio = sum(
        icec_el.xs(electronE,R)*FC_abinitio[vi][vf] for vf in range(7)
    )
    xs_tot_PI_abinitio = sum(
        icec.xs(electronE,R,vi,vf) for vf in range(7)
    )
    xs_tot_FC_Morse = sum(
        icec_FC.xs(electronE,R,vi,vf) for vf in range(icec_FC.Morse_Dp.vmax+1)
    )
    
    print(f'tot from {vi}')
    print(' 1. ab initio PI', xs_tot_PI_abinitio*Units.AU2MB)
    print(' 2. FC ab initio', xs_tot_FC_abinitio*Units.AU2MB)
    print(' ratio 2/1      ', xs_tot_FC_abinitio/xs_tot_PI_abinitio)
    print(' 3. FC Morse    ', xs_tot_FC_Morse*Units.AU2MB)
    print(' ratio 3/1      ', xs_tot_FC_Morse/xs_tot_PI_abinitio)
    print(' ratio 3/2      ', xs_tot_FC_Morse/xs_tot_FC_abinitio)
    
    for vf in range(5):
        print(f'{vi}->{vf}')
        print(' ab initio PI', icec.xs(electronE,R,0,vf)*Units.AU2MB)
        print(' FC ab initio', icec_el.xs(electronE,R)*FC_abinitio[vi][vf]*Units.AU2MB)
        print(' FC Morse    ', icec_FC.xs(electronE,R,0,vf)*Units.AU2MB)
        
    print('\n--- v-ratios ---')
    for vf in range(1,5):
        print(f'{vi}->{vf}/{vi}->{vf-1}')
        print(' PI          ', icec.PI_xs_D(0,vf,omega)/icec.PI_xs_D(0,vf-1,omega))
        print(' FC ab initio', FC_abinitio[vi][vf]/FC_abinitio[vi][vf-1])
        print(' FC Morse    ', icec_FC.FC_factor(0,vf)/icec_FC.FC_factor(0,vf-1))
        
def print_FC_factor(icec:IntraICEC, vD, E):
    fc_factor = icec.FC_factor(0, 0)
    print(f"FC factor 0 -> 0 : {fc_factor}")
    fc_factor = icec.FC_bc_D(vD, E, dps=50)
    print(f"FC factor {vD} -> {round(E*Units.HARTREE2EV,1)} eV : {fc_factor}")
    
def print_boltzmann_probabilities(icec:IntraICEC, T):
    print('\n--- Boltzmann probabilities ---')
    vmax = icec.Morse_D.vmax
    vmax_values = [2,2,8]
    for t,vmax in zip(T,vmax_values):
        print(f'T = {t} K')
        norm = icec.Morse_D.boltzmann_norm(t)
        for vD in range(vmax+1):
            print(f' vi = {vD}: {icec.Morse_D.boltzmann_occupation(t, vD, norm)}')  
            
def print_PI_crosssection(icec:IntraICEC):
    omega = 14.6*Units.EV2HARTREE
    print("\n--- PI cross section at omega = 14.6 eV ---")
    print(f' H    : {icec.PI_xs_A(omega)*Units.AU2MB} Mb')
    print(f' omega: {omega*Units.HARTREE2EV} eV')
    print(f' LiH  : {icec.PI_xs_D_electronic(omega)*Units.AU2MB} Mb')
       
        
# ========= Pre calculate roots of box for dissociative states =============
    
def calculate_roots(Morse:Morse, fname, max_energy:float=1*Units.EV2HARTREE, num:int=500):
    roots, root_estimates = Morse.save_diss_states(fname, max_energy, num)
    plot.config.set_rcParams()
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
    fname = DIR + f'plots/LiHp.roots.E{round(max_energy*Units.HARTREE2EV,1)}eV.{round(Morse.box_length*Units.BOHR2ANGSTROM)}A.pdf'
    fig.savefig(fname)


# ================ Main ===================

# ===== Define which parts are active =====
calc_roots      = 1
bb              = 0
bc              = 1
FC              = 1
plotting        = 1
calculate       = 1
spectra         = 1
cross_section   = 0
temp_dependence = 1
plot_info       = 0
print_info      = 0

electronE   = 1*Units.EV2HARTREE
R           = 6*Units.ANGSTROM2BOHR
L           = 8*Units.ANGSTROM2BOHR
#L = 16*Units.ANGSTROM2BOHR
R_list      = np.array([6,8,10]) * Units.ANGSTROM2BOHR
T           = [15, 300, 1500] 

vD_max_bc   = 7
min_kinE    = 0.01 * Units.EV2HARTREE
max_kinE    = 9 * Units.EV2HARTREE
max_dissE   = 1.8 * Units.EV2HARTREE
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

# --- calculate or load dissociative energies ---
icec_FC.Morse_Dp.define_box(L)
fname = DIR + f'data/LiH/LiHp.diss_energies.E{round(max_dissE*Units.HARTREE2EV,1)}eV.L{round(L*Units.BOHR2ANGSTROM)}A.txt'
if calc_roots:
    calculate_roots(icec_FC.Morse_Dp, fname, max_energy=max_dissE, num=1000)
icec_FC.Morse_Dp.load_diss_states(fname)

# --- electronic ICEC without any nuclear dynamics ---
icec_el = ICEC(*Hp_LiH.input_electronic)
IP_vertical = icec_el.IP_B + (icec.Morse_Dp.V(icec.Morse_D.re) + icec.Morse_Dp.De)
icec_el.IP_B = IP_vertical
icec_el.make_energy_grid(min_kinE*Units.HARTREE2EV, LiH.max_kinE_unresolved*Units.HARTREE2EV, resolution)


energy_diff_at_inf = (7.974721285 - 7.776735464) * Units.HARTREE2EV # energy difference at R=inf
if print_info:
    print('\n===== Info =====')
    print(f"vertical ionization energy {IP_vertical*Units.HARTREE2EV} eV")
    print(f"adiabatic ionizaton energy {IP_adiabatic*Units.HARTREE2EV} eV")
    print(f"energy diff at R=inf       {energy_diff_at_inf} eV")
    #print_FC_factor(icec_FC, 0, 1.3*Units.EV2HARTREE)
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
        if spectra:      
            calculate_spectrum(system, header, icec_el, icec, R, electronE, LiH.v_max, LiHp.v_max)
            if FC:
                calculate_spectrum(system, header, icec_el, icec_FC, R, electronE, modifier='-FC')

    if bc:
        if cross_section:
            calculate_xs_bc(system, header, icec_FC, R, vD_max_bc, modifier='-FC')
        if spectra:
            calculate_spectrum_bc(system, header, icec_FC, R, electronE, vD_max_bc, modifier='-FC')

if plotting:
    if bb:
        if cross_section and FC:
            cross_sections.plot_xs_FC_bb(system, icec_FC, R, icec_el)
        if spectra and FC:
            spectrum.plot_spectrum_FC(system, R, electronE, LiH.v_max, icec_el=icec_el)
        #plot_xs_boltzmann(system, icec, R, T, LiH.v_max, LiH.vib_energies)
    
    if bc:
        if cross_section:
            cross_sections.plot_xs_FC(system, icec_FC, R, icec_el)
        if spectra:   
            spectrum.plot_spectrum_bc(system, icec_FC, R, electronE, vD=0, icec_el=icec_el)
    if cross_section and temp_dependence:
        cross_sections.plot_xs_boltzmann_FC(system, icec_FC, R, T, vD_max_bc, icec_el=icec_el)
    if spectra and temp_dependence:
        spectrum.plot_boltzmann_FC(system, icec_FC, R, electronE, T, vD_max_bc)

if plot_info:
    pes.plot_diss_at_L(icec_FC.Morse_Dp, "LiH", L)
    pes.plot_PES(icec_FC, 'LiH', L, energy_diff_at_inf)
    #plot_H_PI_PR(icec_el)