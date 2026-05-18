import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import DIR_DATA
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.constants import Units
from HLiH.data.H.H import H
from HLiH.data.LiH.LiH import LiH, Li, LiHp
import calc.cross_section
import plot

# ============= Information ===============

def test_FC_factors(icec_el: ICEC, icec:IntraICEC, icec_FC:IntraICEC, R:float):
    electronE = 4*Units.EV2HARTREE
    omega = electronE + icec_el.IP_A
    vi = 0
    
    print('\n--- FC factor ---')
    print('tot b-b from 0')
    FC_abinitio_sum = sum(LiH.FC_abinitio[vi][vf] for vf in range(7))
    FC_Morse_sum = sum(icec_FC.FC_factor(vi,vf) for vf in range(icec_FC.Morse_Dp.vmax+1))
    print(' FC ab initio', FC_abinitio_sum)
    print(' FC Morse    ', FC_Morse_sum)
    print(' ratio       ', FC_Morse_sum/FC_abinitio_sum)
    
    for vf in range(5):
        print(f'{vi}->{vf}')
        print(' PI / PI elec', icec.PI_xs_D(vi,vf,omega)/icec_el.PI_xs_B(omega))
        print(' FC Morse    ', icec_FC.FC_factor(vi,vf))
        print(' FC ab initio', LiH.FC_abinitio[vi][vf])
        
    print(f'\n--- ICEC Cross section [Mb] at {round(electronE*Units.HARTREE2EV,1)} eV---')
    xs_tot_FC_abinitio = sum(
        icec_el.xs(electronE,R)*LiH.FC_abinitio[vi][vf] for vf in range(7)
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
        print(' FC ab initio', icec_el.xs(electronE,R)*LiH.FC_abinitio[vi][vf]*Units.AU2MB)
        print(' FC Morse    ', icec_FC.xs(electronE,R,0,vf)*Units.AU2MB)
        
    print('\n--- v-ratios ---')
    for vf in range(1,5):
        print(f'{vi}->{vf}/{vi}->{vf-1}')
        print(' PI          ', icec.PI_xs_D(0,vf,omega)/icec.PI_xs_D(0,vf-1,omega))
        print(' FC ab initio', LiH.FC_abinitio[vi][vf]/LiH.FC_abinitio[vi][vf-1])
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
    
def print_R_min():
    # min R between H and LiH
    print(f"Hp_LiH.R_min    = {round(Hp_LiH.R_min_vdw,5)} a.u. = {round(Hp_LiH.R_min_vdw*Units.BOHR2ANGSTROM,5)} A")
    R_min_COM = (H.r_vdw + Li.r_vdw + LiH.r_mu)*Units.BOHR2ANGSTROM 
    print(f"R_COM vdW H-LiH = {round(R_min_COM,5)} A")
    R_min_COM = (H.r_vdw + H.r_vdw + (LiH.Req - LiH.r_mu))*Units.BOHR2ANGSTROM 
    print(f"R_COM vdW H-HLi = {round(R_min_COM,5)} A")
    
    
# ============== H+ = LiH =============

class Hp_LiH():
    R_min_vdw = (Li.r_vdw + H.r_vdw + LiH.Req)/2 + H.r_vdw

    input_electronic = [H.deg_factor, H.IP, LiH.IP, H.PI_xs, LiH.PI_xs]
    input_resolved   = [H.deg_factor, H.IP, LiH.IP, H.PI_xs, LiH.file_PI_xs_resolved]
    input_unresolved = [H.deg_factor, H.IP, LiH.IP, H.PI_xs, LiH.file_PI_xs_unresolved]
    
    def R_min():
        # https://doi.org/10.1039/D3CP02959J Tab.3
        r_LiH = 1.646 * Units.ANGSTROM2BOHR
        r_HH = 2.513 * Units.ANGSTROM2BOHR
        r_COM_LiH = (H.m * r_LiH + 0) / (H.m + Li.m)
        R_min = r_LiH - r_COM_LiH + r_HH
        return R_min
    

electronE   = 1*Units.EV2HARTREE
R           = Hp_LiH.R_min() # 6*Units.ANGSTROM2BOHR
L           = 8*Units.ANGSTROM2BOHR
T           = [15, 300, 1500] 

vD_max_bc   = 7
min_kinE    = 0.01 * Units.EV2HARTREE
max_kinE    = 9 * Units.EV2HARTREE
max_dissE   = 2 * Units.EV2HARTREE
max_dissE_1 = 1 * Units.EV2HARTREE
num_grid    = 1000
n_max       = 10     # rydberg states

system = 'Hp-LiH'
title = r'$\text{H}^+ \text{LiH}$'

system = 'Hp-LiH'
header = 'e- + H+ + LiH -> H + LiH+ + e-\n'
    
# --- ICEC with vibrationally resolved photoionization cross section of D ---
icec = IntraICEC(*Hp_LiH.input_resolved)
icec.input_vib_spacing_D(LiH.vib_spacing, LiHp.vib_spacing)
icec.make_energy_grid(min_kinE, max_kinE, num_grid)
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
icec_FC.make_energy_grid(min_kinE, maxEnergy=4.5*Units.EV2HARTREE, num=num_grid)
icec_FC.define_PI_xs_D(method="FC")

# --- calculate or load dissociative energies ---
icec_FC.Morse_Dp.define_box(L)
fname = DIR_DATA + f'LiH/LiHp.diss_energies.E{round(max_dissE*Units.HARTREE2EV,1)}eV.L{round(L*Units.BOHR2ANGSTROM)}A.txt'    
icec_FC.Morse_Dp.load_diss_states(fname)

# --- electronic ICEC without any nuclear dynamics ---
icec_el = ICEC(*Hp_LiH.input_electronic)
IP_vertical = icec_el.IP_B + (icec.Morse_Dp.V(icec.Morse_D.re) + icec.Morse_Dp.De)
icec_el.IP_B = IP_vertical
icec_el.make_energy_grid(min_kinE, LiH.max_kinE_unresolved, num_grid)
    
print('\n===== Info =====')
print(f"vertical ionization energy {round(IP_vertical*Units.HARTREE2EV,3)} eV")
print(f"   approx                  {round(LiH.IP_vert_approx*Units.HARTREE2EV,3)} eV")
print(f"adiabatic ionizaton energy {round(IP_adiabatic*Units.HARTREE2EV,3)} eV")
print(f"   approx min to min       {round(LiH.IP_min_approx*Units.HARTREE2EV,3)} eV")
print(f"energy diff at R=inf       {round(LiH.energy_diff_at_inf*Units.HARTREE2EV,3)} eV")
#print_FC_factor(icec_FC, 0, 1.3*Units.EV2HARTREE)
test_FC_factors(icec_el, icec, icec_FC, R)
print_boltzmann_probabilities(icec_FC, T)
print_PI_crosssection(icec_FC)

calc.cross_section.calculate_ratio_tot_vs_electronic(system, icec_el, R) 

plot.pes.plot_diss_at_L(icec_FC.Morse_Dp, "LiH", L)
plot.pes.plot_PES(icec_FC, 'LiH', L, LiH.energy_diff_at_inf*Units.HARTREE2EV)
H.plot_H_PI_PR(icec_el)