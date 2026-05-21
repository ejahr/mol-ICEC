''' 
Defines functions for calculating ICEC cross sections:
- bb: ICEC cross section vs. incoming electron energy for bound-bound transitions of D.
- rydberg_bb: ICEC cross section vs. incoming electron energy for capturing into Rydberg states of a proton-like atom.
- bc: ICEC cross section vs. incoming electron energy. Includes dissociative states of D+.

Also defines the function calculate_ratio_tot_vs_electronic
'''

import numpy as np
from config import DIR_RESULTS
from icec.icec import ICEC
from icec.intraIcec import IntraICEC, RydbergIntraICEC
from icec.constants import Units
from plot.cross_section import read_results

def extend_header(header:str, R:float=None, electronic=False):
    header += "ICEC cross section\n"
    header += f'R = {round(R*Units.BOHR2ANGSTROM,3)} Angstrom\n' 
    if electronic:
        header += 'E_in [eV] | xs [Mb]'
    return header

def extend_header_bb(header:str, R:float=None, vD_max:int=None, vDp_max:int=None):
    header = extend_header(header, R)
    header += f'Number of initial vibrational states: {vD_max+1}\n'
    header += f'Number of final vibrational states: {vDp_max+1}\n'
    header += 'E_in [eV] | xs vD=0 [Mb]'
    if  vD_max > 0:
        header += f' | ... | xs vD={vD_max+1} [Mb]' 
    return header
    
def extend_header_bc(header:str, icec:IntraICEC, R:float=None, vD_max:int=None, max_dissE=None):
    header = extend_header(header, R)
    header += f'Number of initial vibrational states: {vD_max+1}\n'
    if max_dissE is None:
        max_dissE = icec.Morse_Dp.diss_energies[-1]
    header += "Dissociative states of D+: " + \
    f"Max energy = {round(max_dissE*Units.HARTREE2EV,1)} eV, " + \
    f"Box length = {round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM)} Angstrom\n" 
    header += 'E_in [eV] | xs vD=0 [Mb]'
    if  vD_max > 0:
        header += f' | ... | xs vD={vD_max+1} [Mb]' 
    return header

# ========= Running calculations and saving results ============

def electronic(system:str, header:str, icec:ICEC, R:float):
    # TODO
    header = extend_header(header, R, electronic=True)
    xs_array = icec.energyGrid * Units.HARTREE2EV
    xs = icec.xs_energy(R) * Units.AU2MB
    xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR_RESULTS + f"{system}.xs-electronic.R{round(R*Units.BOHR2ANGSTROM)}.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)

def bb(system:str, header:str, icec:IntraICEC, R:float, vD_max:int=None, vDp_max:int=None, modifier:str=''):
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    if vDp_max is None:
        vDp_max = icec.Morse_Dp.vmax
    header = extend_header_bb(header, R, vD_max, vDp_max)
    xs_array = icec.energyGrid * Units.HARTREE2EV
    for v in range(vD_max+1):
        xs = icec.xs_vD(R, v, vDp_max) * Units.AU2MB
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    # TODO rename to xs.bb.  
    file_path = DIR_RESULTS + f"{system}.xs{modifier}.R{round(R*Units.BOHR2ANGSTROM)}.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)
    
def rydberg_bb(system:str, header:str, icec:RydbergIntraICEC, R:float, n_max:int, vD:int=0, vDp_max:int=None):
    header += f"Rydberg states up to n={n_max} "
    header = extend_header_bb(header, R, vD_max=0, vDp_max=vDp_max)
    xs_array = icec.energyGrid * Units.HARTREE2EV
    for n in range(2, n_max+1):
        icec.n = n
        xs = icec.xs_vD(R, vD, vDp_max) * Units.AU2MB
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR_RESULTS + f"{system}.xs-rydberg.R{round(R*Units.BOHR2ANGSTROM)}.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)
        
def bc(system:str, header:str, icec:IntraICEC, R:float, vD_max:int=None, max_dissE:float=None, modifier:str=''):
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    header = extend_header_bc(header, icec, R, vD_max, max_dissE)
    xs_array = icec.energyGrid*Units.HARTREE2EV
    for vD in range(vD_max+1):
        xs = icec.xs_vD_continuum(R, vD, max_dissE=max_dissE) * Units.AU2MB
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR_RESULTS + f"{system}.xs{modifier}.bc.R{round(R*Units.BOHR2ANGSTROM)}.L{round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM)}.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)    
        
# ===== OTHER =====
    
def calculate_ratio_tot_vs_electronic(system:str, icec_el:ICEC, R:float, vD:int=0, L:float=8*Units.ANGSTROM2BOHR, modifier:str='-FC'):
    results_bb = read_results(system, R, modifier)
    modifier += ".bc"
    results_bc = read_results(system, R, modifier, L)
    results = results_bb[:, vD+1] + results_bc[:, vD+1]
    
    print("\n--- Ratio between total and electronic cross section ---")
    
    for i in [0,600,950]:
        energy = results_bb[i,0]
        xs_tot = results[i]
        xs_el = icec_el.xs(energy*Units.EV2HARTREE, R) * Units.AU2MB
        print(f"electronE    : {round(energy,3)} eV")
        print(f"xs_tot       : {round(xs_tot,3)} MB")
        print(f"xs_el        : {round(xs_el,3)} MB")
        print(f"xs_tot/xs_el : {round(xs_tot/xs_el,5)}")