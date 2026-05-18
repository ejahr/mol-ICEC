import numpy as np
from config import DIR_RESULTS
from icec.intraIcec import IntraICEC, RydbergIntraICEC
from icec.constants import Units

def extend_header(header:str, icec:IntraICEC, R:float=None, vD_max:int=None, vDp_max:int=None, max_dissE=None):
    header += "ICEC cross section\n"
    header += f'R = {round(R*Units.BOHR2ANGSTROM,3)} Angstrom\n' 
    if vD_max is not None:
        header += f'Number of initial vibrational states: {vD_max+1}\n'
    if vDp_max is not None:
        header += f'Number of final vibrational states: {vDp_max+1}\n'
    else:
        max_dissE = max_dissE if max_dissE is not None else icec.Morse_Dp.diss_energies[-1]
        header += "Dissociative states of D+: " + \
        f"Max energy = {round(max_dissE*Units.HARTREE2EV,1)} eV, " + \
        f"Box length = {round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM)} Angstrom\n" 
    header += 'E_in [eV] | xs [Mb]' 
    return header

# ========= Running calculations and saving results ============

def xs_bb(system, header, icec: IntraICEC, R, vD_max=None, vDp_max=None, modifier=''):
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    if vDp_max is None:
        vDp_max = icec.Morse_Dp.vmax
    header = extend_header(header, icec, R, vD_max, vDp_max)
    xs_array = icec.energyGrid * Units.HARTREE2EV
    for v in range(vD_max+1):
        xs = icec.xs_vD(R, v, vDp_max) * Units.AU2MB
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR_RESULTS + f"{system}.xs{modifier}.R{round(R*Units.BOHR2ANGSTROM)}.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)
    
def xs_rydberg_bb(system, header, icec: RydbergIntraICEC, R, n_max, vD=0, vDp_max=None):
    header += f"Rydberg states up to n={n_max} "
    header = extend_header(header, icec, R, vD_max=0, vDp_max=vDp_max)
    xs_array = icec.energyGrid * Units.HARTREE2EV
    for n in range(2, n_max+1):
        icec.n = n
        xs = icec.xs_vD(R, vD, vDp_max) * Units.AU2MB
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR_RESULTS + f"{system}.xs-rydberg.R{round(R*Units.BOHR2ANGSTROM)}.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)
    
def xs_R(system, header, icec, R, vD_max=None, vDp_max=None):
    for r in R:
        xs_bb(system, header, icec, r, vD_max, vDp_max)
    
def xs_bc(system, header, icec: IntraICEC, R, vD_max=None, max_dissE=None, modifier=''):
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    header = extend_header(header, icec, R, vD_max, max_dissE=max_dissE)
    xs_array = icec.energyGrid*Units.HARTREE2EV
    for vD in range(vD_max+1):
        xs = icec.xs_vD_continuum(R, vD, max_dissE=max_dissE) * Units.AU2MB
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR_RESULTS + f"{system}.xs{modifier}.bc.R{round(R*Units.BOHR2ANGSTROM)}.L{round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM)}.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)    
        
def xs_bc_R(system, icec, R, header, vD_max):
    for r in R:
        xs_bc(system, header, icec, r, vD_max)