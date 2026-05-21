''' 
Defines functions for calculating ICEC electron spectra:
- bb: ICEC cross section vs. outgoing electron energy for bound-bound transitions of D.
- bc: ICEC cross section vs. outgoing electron energy for bound-dissociative transitions of D
'''

import numpy as np
from config import DIR_RESULTS
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.constants import Units

def extend_header(header:str, icec:IntraICEC, R:float=None, vD_max:int=None, vDp_max:int=None, electronE:float=None, max_dissE=None):
    header += f"ICEC electron spectrum at E_in = {round(electronE*Units.HARTREE2EV)} eV\n"
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
    return header

# ========= Running calculations and saving results ============

def bb(system: str, header:str, icec_el:ICEC, icec: IntraICEC, R:float, electronE:float, vD_max:int=None, vDp_max:int=None, modifier:str=''): 
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    if vDp_max is None:
        vDp_max = icec.Morse_Dp.vmax
    header = extend_header(header, icec, R, vD_max, vDp_max, electronE)
    header += f"Electronic result: E_out = {round(icec_el.electronE_f(electronE)*Units.HARTREE2EV,5)} eV, xs = {round(icec_el.xs(electronE, R)*Units.AU2MB,5)} Mb\n"
    header += "| v_D, v_Dp, E_out [eV], xs [Mb] |"  
    spectrum_all_vD = np.array([]) 
    for vD in range(vD_max+1):
        spectrum = icec.spectrum(electronE, R, vD, vDp_max)
        if spectrum_all_vD.size == 0:
            spectrum_all_vD = spectrum
        else:
            spectrum_all_vD = np.hstack((spectrum_all_vD, spectrum))
    # TODO rename to spectrum.bb.          
    fname = DIR_RESULTS + f"{system}.spectrum{modifier}.E{round(electronE*Units.HARTREE2EV)}.R{round(R*Units.BOHR2ANGSTROM)}.txt"
    np.savetxt(fname, spectrum_all_vD, fmt='%1.3e', header=header)  
    
def bc(system: str, header: str, icec: IntraICEC, R: float, electronE: float, vD_max:int=None, modifier:str=''): 
    if vD_max is None:
        vD_max = icec.Morse_D.vmax
    header = extend_header(header, icec, R, vD_max, electronE=electronE)
    header += "| vD, diss_energy [eV], E_out [eV], xs [Mb] |"  
    spectrum_all_vD = icec.spectrum_bc(electronE, R, vD=0)  
    for vD in range(1, vD_max+1):
        spectrum = icec.spectrum_bc(electronE, R, vD) 
        spectrum_all_vD = np.hstack((spectrum_all_vD, spectrum))  
    fname = DIR_RESULTS + f'{system}.spectrum{modifier}.bc.E{round(electronE*Units.HARTREE2EV)}.R{round(R*Units.BOHR2ANGSTROM)}.L{round(icec.Morse_Dp.box_length*Units.BOHR2ANGSTROM)}.txt'
    np.savetxt(fname, spectrum_all_vD, fmt='%1.3e', header=header)  