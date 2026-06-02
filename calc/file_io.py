
import os
import numpy as np
from config import DIR_RESULTS, DIR
from icec.constants import Units

def mkdir(dir_name):
    dir_path = os.path.join(DIR, dir_name)
    try:
        os.mkdir(dir_path)
        print(f"Directory '{dir_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{dir_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
# --- cross section ---

def get_fpath_xs(system:str, R:float, modifier:str='', L:float=None):
    fname = f'{system}.xs{modifier}.R{round(R*Units.BOHR2ANGSTROM)}'
    if L is not None:
        fname += f'.L{round(L*Units.BOHR2ANGSTROM)}'
    return os.path.join(DIR_RESULTS, fname + '.txt')

def read_xs(system:str, R:float, modifier:str='', L:float=None):
    'reads in ICEC results'
    fpath = get_fpath_xs(system, R, modifier, L)
    results = np.loadtxt(fpath, comments='#')
    return results
    # TODO
    # energies = results[:,0]
    # xs = results[:,1:]
    # return energies, xs
    
def save_xs(xs_array, header:str, system:str, R:float, modifier:str='', L:float=None):
    'Saves ICEC cross sections. Transposes xs_array before writing.'
    fpath = get_fpath_xs(system, R, modifier, L)
    np.savetxt(fpath, np.transpose(xs_array), fmt='%1.3e', header=header)
    
# --- spectrum ---

def get_fpath_spectrum(system:str, electronE:float, R:float, modifier:str='', L:float=None):
    fname =f"{system}.spectrum{modifier}.E{round(electronE*Units.HARTREE2EV)}.R{round(R*Units.BOHR2ANGSTROM)}"
    if L is not None:
        fname += f'.L{round(L*Units.BOHR2ANGSTROM)}'
    return os.path.join(DIR_RESULTS, fname + '.txt')

def read_spectrum(system:str, electronE:float, R:float, modifier:str='', L:float=None):
    fpath = get_fpath_spectrum(system, electronE, R, modifier, L)
    results = np.loadtxt(fpath, comments='#')
    return results  

def save_spectrum(spectrum, header:str, system:str, electronE:float, R:float, modifier:str='', L:float=None):
    'Saves the ICEC spectrum.'
    fpath = get_fpath_spectrum(system, electronE, R, modifier, L)
    np.savetxt(fpath, spectrum, fmt='%1.3e', header=header)  

def spectrum_idx(vD:int, key:str):
    ''' key: 'v_Dp', 'diss_energy', 'E_out', or 'xs'
    '''
    if key == 'v_Dp':           # final vibrational state, for bound-bound spectra
        return 4*vD+1
    elif key == 'diss_energy':  # final vibrational energy, for bound-dissociative spectra
        return 4*vD+1
    elif key == 'E_out':        # outgoing electron energy, corresponds to electronEf
        return 4*vD+2
    elif key == 'xs':           # icec cross section
        return 4*vD+3
    else:
        raise ValueError(
            "key not recognized, must be 'v_Dp', 'diss_energy', 'E_out', or 'xs'"
        )    