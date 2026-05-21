'''
Old file, similar to run_system.py, but only for H+ LiH
Initiates the calculation of ICEC cross sections and spectra.
Parameters are defined in .config.py
'''

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from HLiH import config
from HLiH.data.LiH.LiH import LiH
from icec.icec import ICEC
from icec.intraIcec import IntraICEC, RydbergIntraICEC
from icec.constants import Units
import calc
import plot

# ============== H+ = LiH =============

electronE           = config.electronE
R                   = config.R 

vD_max_FC           = config.vD_max_FC
min_kinE            = config.min_kinE
min_kinE_rydberg    = 4.35*Units.EV2HARTREE
max_kinE            = config.max_kinE
max_dissE           = config.max_dissE
max_dissE_1         = 1 * Units.EV2HARTREE
num_grid            = config.num_grid
n_max               = config.n_max 

unitA               = config.unitA
unitD               = config.unitD
unitDp              = config.unitDp
system              = config.system_name
header              = config.reaction + '\n'

PI_xs_A = calc.fit.generate_polyfit(unitA.file_PI_xs, unitA.degree)

morse_parameters_initial = (unitD.mu, unitD.we, unitD.Req, unitD.De, unitD.wexe) 
morse_parameters_final = (unitDp.mu, unitDp.we, unitDp.Req, unitDp.De, unitDp.wexe)

xs_data = np.loadtxt(unitD.file_PI_xs_unresolved, comments='#')
energies = xs_data[:,0]
max_kinE_unresolved = energies[-1]*Units.EV2HARTREE - unitA.IP

# ===== Initialize ICEC classes =====

# --- ICEC with vibrationally resolved photoionization cross section of D ---
icec = IntraICEC(unitA.deg_factor, unitA.IP, unitD.IP, PI_xs_A, unitD.file_PI_xs_resolved)
icec.input_vib_spacing_D(unitD.vib_spacing, unitDp.vib_spacing)
icec.define_Morse_D(*morse_parameters_initial)
icec.define_Morse_Dp(*morse_parameters_final)
icec.change_minima_to_adiabatic_IP()
icec.make_energy_grid(min_kinE, max_kinE, num_grid)
icec.define_PI_xs_D(method="resolved")

# --- Rydberg ---
icec_rydberg : RydbergIntraICEC = RydbergIntraICEC.from_IntraICEC(icec, n=2)
icec_rydberg.make_energy_grid(min_kinE_rydberg, max_kinE, int(num_grid/2))

# --- ICEC with Franck-Condon model ---
icec_FC = IntraICEC(unitA.deg_factor, unitA.IP, unitD.IP, PI_xs_A, unitD.file_PI_xs_unresolved)
icec_FC.define_Morse_D(*morse_parameters_initial)
icec_FC.define_Morse_Dp(*morse_parameters_final, box_length=config.L)
icec_FC.change_minima_to_adiabatic_IP()
icec_FC.make_energy_grid(min_kinE, max_kinE_unresolved, num_grid)
icec_FC.define_PI_xs_D(method="FC")

# --- calculate or load dissociative energies ---
if config.FC and config.bc:
    fname = config.DIR_DATA + f'{unitD.name}/{unitDp.name}.diss_energies.E{round(max_dissE*Units.HARTREE2EV,1)}eV.L{round(config.L*Units.BOHR2ANGSTROM)}A.txt'
    if config.calc_roots:
        roots, root_estimates = icec_FC.Morse_Dp.save_diss_states(fname, max_dissE, num=1000)
        plot.pes.roots(icec_FC.Morse_Dp, roots, root_estimates, max_dissE)
    icec_FC.Morse_Dp.load_diss_states(fname)

# --- electronic ICEC without any nuclear dynamics ---
PI_xs_D_el      = calc.fit.generate_linfit(unitD.file_PI_xs_unresolved)
icec_el         = ICEC(unitA.deg_factor, unitA.IP, unitD.IP, PI_xs_A, PI_xs_D_el)
icec_el.IP_D    = icec.convert_minima_to_vertical_IP(unitD.IP)
icec_el.make_energy_grid(min_kinE, max_kinE_unresolved, num_grid)

# ===== CALCULATION =====

if config.calculate:
    if config.bb: 
        if config.cross_section:
            if config.resolved:
                print('--- b-b xs resolved ---')
                calc.cross_section.bb(system, header, icec, R, unitD.v_max, unitDp.v_max)
            if config.FC:
                print('--- b-b xs FC ---')
                calc.cross_section.bb(system, header, icec_FC, R, vD_max_FC, modifier='-FC')
            if config.rydberg:
                print('--- b-b xs rydberg ---')
                calc.cross_section.rydberg_bb(system, header, icec_rydberg, R, n_max, 0, unitDp.v_max)
                
        if config.spectra: 
            if config.resolved:   
                print('--- b-b spectrum resolved ---')  
                calc.spectrum.bb(system, header, icec_el, icec, R, electronE, unitD.v_max, unitDp.v_max)
            if config.FC:
                print('--- b-b spectrum FC ---')  
                calc.spectrum.bb(system, header, icec_el, icec_FC, R, electronE, vD_max_FC, modifier='-FC')

    if config.bc and config.FC:
        if config.cross_section:
            print('--- b-c xs FC ---')
            calc.cross_section.bc(system, header, icec_FC, R, vD_max_FC, max_dissE=max_dissE, modifier='-FC')
        if config.spectra:
            print('--- b-c spectrum FC ---')
            calc.spectrum.bc(system, header, icec_FC, R, electronE, vD_max_FC, modifier='-FC')

# ===== PLOTS =====

if config.plotting:
    if config.bb:
        if config.cross_section and config.FC:
            plot.cross_section.bb_resolved_FC_rydberg(system, icec, R, icec_el)
        if config.spectra and config.FC:
            plot.spectrum.bb_resolved_and_FC(system, R, electronE, unitD.v_max, icec_el=icec_el)
    
    if config.bc and config.FC:
        if config.cross_section:
            plot.cross_section.bb_and_bc(system, icec_FC, R, icec_el)
        if config.spectra:   
            plot.spectrum.bb_and_bc(system, icec_FC, R, electronE, vD=0, icec_el=icec_el, secax_label=r"$E_+$ [eV]")
            
    if config.spectra and config.temp_dependence:
        plot.spectrum.boltzmann_bb_and_bc(system, icec_FC, R, electronE, config.T, vD_max_FC)


plot.pes.pes(icec_FC, 'LiH', config.L, LiH.energy_diff_at_inf*Units.HARTREE2EV)