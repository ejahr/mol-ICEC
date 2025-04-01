import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#sys.path.insert(0, '/mnt/home/elena/icec_project') % on linux servers
sys.path.insert(0, '/home/elena/icec-project')
from crosssection.icec.constants import *

DIR = '/home/elena/icec-project/dimers/'

def energy_to_hartree(energies, unit="eV"):
    if unit == "eV":
        return energies * EV2HARTREE
    elif unit == "A":
        return angstrom2hartree(energies)
    else:
        print('unit not found. Try eV, A')

def generate_polyfit(fname, degree, energy_unit="eV"):
    xs_data = np.loadtxt(fname, comments='#')
    energies = energy_to_hartree(xs_data[:,0], energy_unit)
    xs = xs_data[:,1] * MB2AU
    coefficients = np.polyfit(energies, xs, degree)
    return np.poly1d(coefficients)

def test_polyfit(polyfit, x, y, fname):
    x_fit = np.linspace(min(x), 20*EV2HARTREE, 100)
    y_fit = polyfit(x_fit)
    plt.xlim(min(x)*HARTREE2EV, 20)
    plt.scatter(x*HARTREE2EV, y*AU2MB, color='red', label='Data points')  # Original data points
    plt.plot(x_fit*HARTREE2EV, y_fit*AU2MB, label=f'Polynomial fit')  # Fitted curve
    plt.legend()
    plt.savefig(fname)
    
def generate_linfit(fname, energy_unit="eV"):
    xs_data = np.loadtxt(fname, comments='#')
    energies = energy_to_hartree(xs_data[:,0], energy_unit)
    xs = xs_data[:,1] * MB2AU
    interp_func = sp.interpolate.interp1d(energies, xs, kind='linear', fill_value="extrapolate")
    return interp_func

def test_linfit(interp_func, x, y, fname):
    x_fit = np.geomspace(min(x), 20*EV2HARTREE, 1000)
    y_fit = interp_func(x_fit)
    plt.xlim(min(x)*HARTREE2EV, 20)
    plt.scatter(x*HARTREE2EV, y*AU2MB, color='red', label='Data points') 
    plt.plot(x_fit*HARTREE2EV, y_fit*AU2MB, label=f'Linear fit')  
    plt.legend()
    plt.savefig(fname)