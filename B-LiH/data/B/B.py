import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from calc.fit import generate_linfit, generate_chebyshev
from icec.constants import Units

# =================== B+ ==========================
class B:
    deg_2P = 6
    deg_1S = 1
    deg_factor = deg_2P/deg_1S
    IP = 8.298019 * Units.EV2HARTREE

    PI_xs_fname = os.path.dirname(os.path.realpath(__file__)) + '/B_PI-xs.txt'
    PI_xs_linear = generate_linfit(PI_xs_fname)
    PI_xs = generate_chebyshev(PI_xs_fname)
     

if __name__ == "__main__":
    data = np.loadtxt(B.PI_xs_fname, comments='#')
    x = data[:,0] * Units.EV2HARTREE
    y = data[:,1] * Units.MB2AU
    print(min(x))
    x_fit = np.geomspace(min(x), 20*Units.EV2HARTREE, 1000)
    y_fit = B.PI_xs(x_fit)
    plt.yscale('log')
    plt.xlim(min(x)*Units.HARTREE2EV, 20)
    plt.scatter(x*Units.HARTREE2EV, y*Units.AU2MB, color='red', label='Data points') 
    plt.plot(x_fit*Units.HARTREE2EV, y_fit*Units.AU2MB, label=f'Linear fit')  
    plt.legend()
    fname = os.path.dirname(os.path.realpath(__file__)) + '/test_linfit.pdf'
    plt.savefig(fname)
