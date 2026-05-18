from HLiH.data.H.H import H
from HLiH.data.LiH.LiH import LiH, Li, LiHp

from icec.icec import ICEC
from icec.intraIcec import IntraICEC, RydbergIntraICEC
from icec.morse import Morse
from icec.constants import Units

import calc
import plot
import config

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

# ===== Define which parts are active =====
calc_roots      = 0
bb              = 1
bc              = 0
FC              = 1
plotting        = 0
calculate       = 1
spectra         = 1
cross_section   = 0
temp_dependence = 0
plot_info       = 1
print_info      = 0

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

# ===== Define ICEC classes for calculating ICEC cross sections =====

# --- ICEC with vibrationally resolved photoionization cross section of D ---
icec = IntraICEC(*Hp_LiH.input_resolved)
icec.input_vib_spacing_D(LiH.vib_spacing, LiHp.vib_spacing)
icec.make_energy_grid(min_kinE, max_kinE, num_grid)
icec.define_Morse_D(*LiH.morse_parameters, wexe=LiH.wexe)
icec.define_Morse_Dp(*LiHp.morse_parameters, wexe=LiHp.wexe)
IP_adiabatic = icec.IP_D - (icec.Morse_D.energy(0)+ icec.Morse_D.De) + (icec.Morse_Dp.energy(0)+ icec.Morse_Dp.De)
icec.IP_D = IP_adiabatic
icec.define_PI_xs_D(method="resolved")

# --- Rydberg ---
icec_rydberg : RydbergIntraICEC = RydbergIntraICEC.from_IntraICEC(icec, n=2)
icec_rydberg.make_energy_grid(4.35*Units.EV2HARTREE, max_kinE, int(num_grid/2))

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
if calc_roots:
    roots, root_estimates = Morse.save_diss_states(fname, max_dissE, num=1000)
    plot.pes.plot_roots(icec_FC.Morse_Dp, roots, root_estimates, max_dissE)
    
icec_FC.Morse_Dp.load_diss_states(fname)

# --- electronic ICEC without any nuclear dynamics ---
icec_el = ICEC(*Hp_LiH.input_electronic)
IP_vertical = icec_el.IP_B + (icec.Morse_Dp.V(icec.Morse_D.re) + icec.Morse_Dp.De)
icec_el.IP_B = IP_vertical
icec_el.make_energy_grid(min_kinE, LiH.max_kinE_unresolved, num_grid)

if print_info:
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
    
    plot.cross_section.calculate_ratio_tot_vs_electronic(system, icec_el, R) 

system = 'Hp-LiH'
header = 'e- + H+ + LiH -> H + LiH+ + e-\n'

if calculate:
    if bb:
        if cross_section:
            #calc.cross_section.xs_bb(system, header, icec, R, LiH.v_max, LiHp.v_max)
            calc.cross_section.xs_rydberg_bb(system, header, icec_rydberg, R, n_max, 0, LiHp.v_max)
            #if FC:
                #calc.cross_section.xs_bb(system, header, icec_FC, R, modifier='-FC')
        if spectra:      
            calc.spectrum.spectrum_bb(system, header, icec_el, icec, R, electronE, LiH.v_max, LiHp.v_max)
            if FC:
                calc.spectrum.spectrum_bb(system, header, icec_el, icec_FC, R, electronE, modifier='-FC')

    if bc:
        if cross_section:
            calc.cross_section.xs_bc(system, header, icec_FC, R, vD_max=0, max_dissE=max_dissE_1, modifier='-FC')
        if spectra:
            calc.spectrum.spectrum_bc(system, header, icec_FC, R, electronE, vD_max_bc, modifier='-FC')

if plotting:
    if bb:
        if cross_section and FC:
            plot.cross_section.xs_FC_bb(system, icec, R, icec_el)
        if spectra and FC:
            plot.spectrum.spectrum_FC_bb(system, R, electronE, LiH.v_max, icec_el=icec_el)
    
    if bc:
        if cross_section:
            plot.cross_section.xs_FC(system, icec_FC, R, icec_el)
        if spectra:   
            plot.spectrum.spectrum_FC(system, icec_FC, R, electronE, vD=0, icec_el=icec_el, secax_label=r"$E_{\mathrm{LiH}^+}$ [eV]")
    #if cross_section and temp_dependence:
    #    cross_sections.plot_xs_boltzmann_FC(system, icec_FC, R, T, vD_max_bc, icec_el=icec_el)
    if spectra and temp_dependence:
        plot.spectrum.boltzmann_FC(system, icec_FC, R, electronE, T, vD_max_bc)
