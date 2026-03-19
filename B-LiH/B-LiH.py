from data.B.B import B
from data.LiH.LiH import LiH, LiHp

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from icec.intraIcec import IntraICEC
from icec.constants import Units

import calc
import plot

DIR = os.path.dirname(os.path.realpath(__file__)) + '/'

system = 'Bp-LiH'
header = 'e- + B+ + LiH -> B + LiH+ + e-\n'

calculate   = 0

# ===================== B+ = LiH =================
input       = [B.deg_factor, B.IP, LiH.IP, B.PI_xs, LiH.file_PI_xs_resolved]
R           = 6 * Units.ANGSTROM2BOHR
min_kinE    = 0.1 * Units.EV2HARTREE
max_kinE    = 15 * Units.EV2HARTREE
num_grid    = 500

# --- ICEC with vibrationally resolved photoionization cross section of D ---
icec = IntraICEC(*input)

icec.define_PI_xs_D(method="resolved")

icec.define_Morse_D(*LiH.morse_parameters, wexe=LiH.wexe)
icec.define_Morse_Dp(*LiHp.morse_parameters, wexe=LiHp.wexe)

icec.make_energy_grid(min_kinE, max_kinE, num_grid)

# --- calculate Cross section ----
if calculate:
    header = calc.cross_section.extend_header(header, icec, R, LiH.v_max, LiHp.v_max)
    xs_array = icec.energyGrid * Units.HARTREE2EV
    for v in range(LiH.v_max+1):
        xs = icec.xs_vD(R, v, LiHp.v_max) * Units.AU2MB
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR + f"results/{system}.xs.R{round(R*Units.BOHR2ANGSTROM)}.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header) 

# --- plot cross section ---
def read_results_file(system, R):
    file_path = DIR + f"results/{system}.xs.R{round(R*Units.BOHR2ANGSTROM)}.txt"
    results = np.loadtxt(file_path, comments='#')
    energies = results[:,0]
    xs = results[:,1:]
    return energies, xs

fig = plt.figure()
ax = plt.gca() 
plot.cross_section.set_axes(ax)

vD = 0
energies, xs = read_results_file(system, R)
ax.plot(energies, xs[:, vD], label=r'b-b')

data = np.loadtxt(B.PI_xs_fname, comments='#')
x = data[:,0] 
y = data[:,1]
ax.plot(x, y, color='blue', label='PI', marker='x') 
ax.plot(x-B.IP * Units.HARTREE2EV, y, color='red', label='PI', marker='x') 

x_e = x - B.IP * Units.HARTREE2EV
x_e = min(x_e[x_e>0])
pr_xs = icec.PR_xs_A(x_e * Units.EV2HARTREE) * Units.AU2MB
ax.scatter(x_e, pr_xs)

icec.make_energy_grid(x_e*Units.EV2HARTREE, max_kinE, num_grid)
icec.plot_PR_xs_A(ax, label=r'$\sigma_\text{PR}$', color='dimgray') 
ax.plot(x, icec.PI_xs_A(x*Units.EV2HARTREE)*Units.AU2MB, label="PI")

#ax.set_yscale('linear')
ax.set_xlim(0,20)
#ax.set_ylim(0,20)
ax.legend()
fname = DIR + f'plots/{system}.xs.v0.R{round(R*Units.BOHR2ANGSTROM)}.pdf'
plt.tight_layout()
fig.savefig(fname)
