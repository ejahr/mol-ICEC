from B import B

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.LiH.LiH import LiH, LiHp

from icec.intraIcec import IntraICEC
from icec.constants import Units

import calc
import plot

DIR = os.path.dirname(os.path.realpath(__file__)) + '/'

system = 'Bp-LiH'
header = 'e- + B+ + LiH -> B + LiH+ + e-\n'

# ===================== B+ = LiH =================
input       = [B.deg_factor, B.IP, LiH.IP, B.PI_xs, LiH.file_PI_xs_resolved]
R           = 6 * Units.ANGSTROM2BOHR
min_kinE    = 0.01 * Units.EV2HARTREE
max_kinE    = 9 * Units.EV2HARTREE
num_grid    = 100

# --- ICEC with vibrationally resolved photoionization cross section of D ---
icec = IntraICEC(*input)

icec.define_PI_xs_D(method="resolved")
icec.input_vib_spacing_D(LiH.vib_spacing, LiHp.vib_spacing)

icec.define_Morse_D(*LiH.morse_parameters, wexe=LiH.wexe)
icec.define_Morse_Dp(*LiHp.morse_parameters, wexe=LiHp.wexe)

icec.make_energy_grid(min_kinE, max_kinE, num_grid)

# --- calculate Cross section ----
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
icec.plot_PR_xs_A(ax, label=r'$\sigma_\text{PR}$', color='dimgray', ls=':') 

ax.legend()
fname = DIR + f'plots/{system}.xs.v0.R{round(R*Units.BOHR2ANGSTROM)}.pdf'
plt.tight_layout()
fig.savefig(fname)
