import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from icec.constants import Units

DIR = '/home/elena/intraICEC/dimers/'

def generate_polyfit(fname, degree):
    xs_data = np.loadtxt(fname, comments='#')
    E_photon = xs_data[:,0] * Units.EV2HARTREE
    xs = xs_data[:,1] * Units.MB2AU
    coefficients = np.polyfit(E_photon, xs, degree)
    #test_polyfit(np.poly1d(coefficients), E_photon, xs, degree)
    return np.poly1d(coefficients)

def test_polyfit(polyfit, x, y, degree):
    x_fit = np.linspace(min(x), 20*Units.EV2HARTREE, 100)
    y_fit = polyfit(x_fit)
    plt.xlim(min(x)*Units.HARTREE2EV, 20)
    plt.scatter(x*Units.HARTREE2EV, y*Units.AU2MB, color='red', label='Data points')  # Original data points
    plt.plot(x_fit*Units.HARTREE2EV, y_fit*Units.AU2MB, label=f'Polynomial fit, degree = ' + str(degree))  # Fitted curve
    plt.legend()
    fname = DIR + 'plot/test_polyfit.pdf'
    plt.savefig(fname)
    
def generate_linfit(fname):
    xs_data = np.loadtxt(fname, comments='#')
    energies = xs_data[:,0] * Units.EV2HARTREE
    xs = xs_data[:,1] * Units.MB2AU
    interp_func = sp.interpolate.interp1d(energies, xs, kind='linear', fill_value="extrapolate")
    #test_linfit(interp_func, energies, xs)
    return interp_func

def test_linfit(interp_func, x, y):
    x_fit = np.geomspace(min(x), 20*Units.EV2HARTREE, 1000)
    y_fit = interp_func(x_fit)
    plt.xlim(min(x)*Units.HARTREE2EV, 20)
    plt.scatter(x*Units.HARTREE2EV, y*Units.AU2MB, color='red', label='Data points') 
    plt.plot(x_fit*Units.HARTREE2EV, y_fit*Units.AU2MB, label=f'Linear fit')  
    plt.legend()
    fname = DIR + 'plot/test_linfit.pdf'
    plt.savefig(fname)
