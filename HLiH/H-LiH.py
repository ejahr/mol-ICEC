'''
Old file which is only for H+ LiH
Similar to run_system.py
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from HLiH.data.H.H import H
from HLiH.data.LiH.LiH import LiH, Li, LiHp

from icec.icec import ICEC
from icec.intraIcec import IntraICEC, RydbergIntraICEC
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

electronE   = config.electronE
R           = Hp_LiH.R_min() # 6*Units.ANGSTROM2BOHR TODO change to config

vD_max_bc   = config.vD_max_bc
min_kinE    = config.min_kinE
max_kinE    = config.max_kinE
max_dissE   = config.max_dissE
max_dissE_1 = 1 * Units.EV2HARTREE
num_grid    = config.num_grid
n_max       = config.n_max     # rydberg states

system      = config.system_name
header      = config.reaction + '\n'
title       = config.title

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
icec_FC.define_Morse_Dp(*LiHp.morse_parameters, wexe=LiHp.wexe, box_length=config.L)
icec_FC.make_energy_grid(min_kinE, maxEnergy=4.5*Units.EV2HARTREE, num=num_grid)
icec_FC.define_PI_xs_D(method="FC")

# --- calculate or load dissociative energies ---
fname = config.DIR_DATA + f'LiH/LiHp.diss_energies.E{round(max_dissE*Units.HARTREE2EV,1)}eV.L{round(config.L*Units.BOHR2ANGSTROM)}A.txt'
if config.calc_roots:
    roots, root_estimates = icec_FC.Morse_Dp.save_diss_states(fname, max_dissE, num=1000)
    plot.pes.plot_roots(icec_FC.Morse_Dp, roots, root_estimates, max_dissE)
    
icec_FC.Morse_Dp.load_diss_states(fname)

# --- electronic ICEC without any nuclear dynamics ---
icec_el = ICEC(*Hp_LiH.input_electronic)
IP_vertical = icec_el.IP_D + (icec.Morse_Dp.V(icec.Morse_D.re) + icec.Morse_Dp.De)
icec_el.IP_D = IP_vertical
icec_el.make_energy_grid(min_kinE, LiH.max_kinE_unresolved, num_grid)

if config.calculate:
    if config.bb:
        if config.cross_section:
            calc.cross_section.xs_bb(system, header, icec, R, LiH.v_max, LiHp.v_max)
            calc.cross_section.xs_rydberg_bb(system, header, icec_rydberg, R, n_max, 0, LiHp.v_max)
            if config.FC:
                calc.cross_section.xs_bb(system, header, icec_FC, R, modifier='-FC')
        if config.spectra:      
            calc.spectrum.spectrum_bb(system, header, icec_el, icec, R, electronE, LiH.v_max, LiHp.v_max)
            if config.FC:
                calc.spectrum.spectrum_bb(system, header, icec_el, icec_FC, R, electronE, modifier='-FC')

    if config.bc:
        if config.cross_section:
            calc.cross_section.xs_bc(system, header, icec_FC, R, vD_max=0, max_dissE=max_dissE_1, modifier='-FC')
        if config.spectra:
            calc.spectrum.spectrum_bc(system, header, icec_FC, R, electronE, vD_max_bc, modifier='-FC')

if config.plotting:
    if config.bb:
        if config.cross_section and config.FC:
            plot.cross_section.xs_FC_bb(system, icec, R, icec_el)
        if config.spectra and config.FC:
            plot.spectrum.spectrum_FC_bb(system, R, electronE, LiH.v_max, icec_el=icec_el)
    
    if config.bc:
        if config.cross_section:
            plot.cross_section.xs_FC(system, icec_FC, R, icec_el)
        if config.spectra:   
            plot.spectrum.spectrum_FC(system, icec_FC, R, electronE, vD=0, icec_el=icec_el, secax_label=r"$E_{\mathrm{LiH}^+}$ [eV]")
    #if config.cross_section and config.temp_dependence:
    #    config.cross_sections.plot_xs_boltzmann_FC(system, icec_FC, R, config.T, vD_max_bc, icec_el=icec_el)
    if config.spectra and config.temp_dependence:
        plot.spectrum.boltzmann_FC(system, icec_FC, R, electronE, config.T, vD_max_bc)
