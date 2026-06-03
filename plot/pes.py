''' 
Defines functions for generating plots:
- pes: potential energy surfaces of D and D+ (currently tailored to LiH -> LiH+) 
- roots: for checking if all discretized solutions for dissociative states are found
- energy_sketch: schematics for ICEC
'''

import os
import mpmath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from icec.intraIcec import IntraICEC
from icec.morse import Morse
from icec.constants import Units
from config import DIR_PLOTS, unitDp
import plot.cfg_plot as cfg_plot

cfg_plot.set_rcParams()

# ===== HELPER FUNCTIONS =====

def plot_vib_state(ax, morse:Morse, vi, scale=1./15, yshift=0):
    psi = [morse.psi(vi,r_i) * scale
               + morse.energy(vi)*Units.HARTREE2EV + yshift 
               for r_i in morse.r]
    ax.plot(morse.r*Units.BOHR2ANGSTROM, psi, color='tab:blue', lw=1)
    
def plot_diss_state(ax, morse:Morse, energy, norm=None, scale=1, yshift=0, color='lightskyblue'):
    if norm is None:
        norm = morse.get_norm_diss(energy)
    psi_diss = [mpmath.re(norm * morse.psi_diss(energy,r_i)) * scale
                + energy*Units.HARTREE2EV + yshift 
                for r_i in morse.r]
    ax.plot(morse.r*Units.BOHR2ANGSTROM, psi_diss, color=color, lw=1)
    
def add_vertical_arrow(ax, x:float, y:tuple[float], text:str="", shift_text_x=0.1, y_text=None, shift_text_y=0):
    arrowstyle = patches.ArrowStyle("<|-|>", head_width=0.1, head_length=0.3)
    arrowprops = dict(arrowstyle=arrowstyle, lw=1, color='tab:red', capstyle='butt', joinstyle="miter")
    ax.annotate(
        "",
        xy=(x, y[0]),        
        xytext=(x, y[1]),
        arrowprops=arrowprops
    )
    if y_text is None:
        y_text = (y[0] + y[1]) / 2 + shift_text_y
    ax.text(
        x + shift_text_x,
        y_text,
        text,
        va='center',
        color='tab:red',
    )
    
def add_cut_out_lines(ax1, ax2):
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    
# ===== PES =====
    
def pes(icec:IntraICEC, system:str, L=5*Units.ANGSTROM2BOHR, yshift:float=0):
    """ 
    Generates plot:
        PES of the ground electronic states of D and D+
        currently adapted to LiH
            
    Args:
        icec: class instance of IntraICEC, defines and calculates all necessary quantities
        system: name of the system 
        L: maximum distance, length of the box
        yshif: difference between the two PES at R -> infty
    """   
    height_ratios           = cfg_plot.height_ratios
    scale                   = cfg_plot.scale
    show_continuum_states   = cfg_plot.show_continuum_states
    num_final_bound_states  = cfg_plot.num_final_bound_states
    yshift                  = cfg_plot.yshift if yshift == 0 else yshift
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=height_ratios, figsize=(5,5))
    fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(bottom=False)
    
    r = icec.Morse_D.make_rgrid(rmax=L)
    icec.Morse_Dp.r = r
    L = L*Units.BOHR2ANGSTROM
    
    # ----- D -----
    ax2.set_xlabel(r'$R$ [$\mathrm{\AA}$]')
    ax2.set_ylim(-icec.Morse_D.De*Units.HARTREE2EV - 0.1, 0.1)
    
    for vi in range(3):
        plot_vib_state(ax2, icec.Morse_D, vi, scale)
        
    V = icec.Morse_D.V(r)
    ax2.plot(r*Units.BOHR2ANGSTROM, V*Units.HARTREE2EV, color='black')
    ax2.annotate(system, (r[-200]*Units.BOHR2ANGSTROM, V[-100]*Units.HARTREE2EV - 0.25))
    
    # vertical line at minimum of PES
    x = np.array([icec.Morse_D.re, r[-1]-0.25]) * Units.BOHR2ANGSTROM
    y = np.array([icec.Morse_D.V(icec.Morse_D.re), icec.Morse_D.V(icec.Morse_D.re)]) * Units.HARTREE2EV - 0.002
    ax2.plot(x, y, ls='--', color='grey', lw=1, zorder=0)
    
    # dissociation energy
    y = [icec.Morse_D.V(icec.Morse_D.re) * Units.HARTREE2EV - 0.01, 0]
    add_vertical_arrow(ax2, x=L-0.2, y=y, text=r"$D_\mathrm{e}$", shift_text_x=0.06, y_text=-1)
    # bound vibrational energy
    y = [icec.Morse_D.energy(2) * Units.HARTREE2EV - 0.01, 0]
    add_vertical_arrow(ax2, x=L-0.37, y=y, text=r"$E_\nu$", shift_text_x=-0.5, y_text=-1)
    # adiabtic IP
    print((icec.Morse_Dp.De + icec.Morse_Dp.energy(0)) * Units.HARTREE2EV)
    y = [icec.Morse_D.energy(0) * Units.HARTREE2EV - 0.01, (icec.Morse_Dp.De + icec.Morse_Dp.energy(0))*Units.HARTREE2EV + 0.6]
    add_vertical_arrow(ax2, x=L/2+0.7, y=y, text=r"$\mathrm{IP}^\mathrm{a}$", shift_text_x=0.06, y_text=-1)
    # vertical IP
    y = [icec.Morse_D.V(icec.Morse_D.re) * Units.HARTREE2EV, (icec.Morse_Dp.De + icec.Morse_Dp.V(icec.Morse_D.re))*Units.HARTREE2EV + yshift + 0.6]
    add_vertical_arrow(ax2, x=icec.Morse_D.re*Units.BOHR2ANGSTROM, y=y, text=r"$\mathrm{IP}^\mathrm{v}$", shift_text_x=0.06, y_text=-1)
    # difference in V(R->oo)
    y = [0, yshift + icec.Morse_Dp.De * Units.HARTREE2EV + 0.5]
    add_vertical_arrow(ax2, x=L-0.2, y=y, text=r"$V^\infty_+ - V^\infty$", shift_text_x=-1.75, shift_text_y=-0.035)

    # ----- D+ -----
    height = height_ratios[0] / height_ratios[1] * (icec.Morse_D.De*Units.HARTREE2EV + 2*0.1)
    y_min = yshift - icec.Morse_Dp.De*Units.HARTREE2EV - 0.1
    ax1.set_ylim(y_min, y_min + height)
    
    # states
    if show_continuum_states:
        energy, norm = icec.Morse_Dp.diss_energies[30], icec.Morse_Dp.diss_norms[30]
        plot_diss_state(ax1, icec.Morse_Dp, energy, norm, scale, yshift)
    for vf in range(num_final_bound_states):
        plot_vib_state(ax1, icec.Morse_Dp, vf, scale, yshift)
    
    V = icec.Morse_Dp.V(r)
    ax1.plot(r*Units.BOHR2ANGSTROM, V*Units.HARTREE2EV + yshift, color='black')
    ax1.annotate(system + '+', (r[-200]*Units.BOHR2ANGSTROM, V[-100]*Units.HARTREE2EV + yshift + 0.1))
    
    # dissociative vibrational energy
    if show_continuum_states:
        add_vertical_arrow(ax1, x=7.8, y1=yshift, y2=energy*Units.HARTREE2EV+yshift+0.02, text=r"$E$", shift_text_x=0.05)
    
    add_cut_out_lines(ax1, ax2)
    
    fig.text(0, 0.5, r'$E-V^\infty$ [eV]', va='center', rotation='vertical')
    fname = os.path.join(
        DIR_PLOTS,
        f"{system}.PES.L{round(L)}.pdf"
    )
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.2)

# ===== INFORMATION PLOTS =====

def roots(morse:Morse, roots, root_estimates, max_energy):
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_title('Dissociative states')
    ax.set_xlabel(r'$E$ [eV]')
    ax.set_ylabel(r'$E$ [a.u.]')
    ax.set_yscale('log')
    ax.bar(root_estimates*Units.HARTREE2EV, root_estimates/2, width=0.005, color='tab:blue', label='estimates')
    ax.bar(roots*Units.HARTREE2EV, roots, width=0.005, color='tab:red', label='roots')
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(
        DIR_PLOTS,
        f'{unitDp.name}.roots.E{round(max_energy*Units.HARTREE2EV)}.L{round(morse.box_length*Units.BOHR2ANGSTROM)}.pdf'
    )
    fig.savefig(fname)

def diss_at_L(morse:Morse, system, L=8*Units.ANGSTROM2BOHR):
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    def psi_at_L(E):
        return mpmath.re(morse.psi_diss(E,L))
    x = np.geomspace(1e-5, 1, 2000) 
    y = np.array([psi_at_L(E*Units.EV2HARTREE) for E in x])

    ax.plot(x,y)
    ax.grid(True)
    ax.set_xlabel(r'$E$ [$\mathrm{eV}$]')
    ax.set_ylabel(r'$\psi_E(L)$')
    ax.set_ylim(-1,1)
    fname = os.path.join(
        DIR_PLOTS,
        f"{system}.psi_at_L{round(L*Units.BOHR2ANGSTROM)}.pdf"
    )
    fig.savefig(fname, bbox_inches='tight')
    
# ===== ICEC ENERGY SKETCH =====
    
def energy_sketch():
    x_D = 1
    y_D = 0.3
    shift_D = 2
    x_A = -1
    shift_A = 2
    
    a = 5
    
    def potD(x):
        return y_D + a*(x-x_D)**2
    
    def potD_inv(y):
        root = np.sqrt((y-y_D)/a)
        return x_D - root, x_D + root
    
    def potDp(x):
        return y_D + shift_D + a*(x-x_D)**2
    
    def potA(x):
        return a*(x-x_A)**2
    
    def potAp(x):
        return shift_A + a*(x-x_A)**2
    
    def potA_inv(y):
        root = np.sqrt(y/a)
        return x_A - root, x_A + root
    
    
    E_D = y_D + 0.3
    E_Dp = E_D + shift_D
    
    E_A = 0.3
    E_Ap = E_A + shift_A

    fig, ax = plt.subplots(figsize=(6,3))
    
    # Arrow
    arrowstyle = patches.ArrowStyle("-|>", head_width=0.12, head_length=0.35)
    arrowprops=dict(arrowstyle=arrowstyle, lw=1.5, color='tab:red', capstyle='butt', joinstyle="miter", zorder=0)
    ax.annotate(
        "",
        xytext=(x_A, shift_A+E_A+0.025),
        xy=(x_A, 0+E_A-0.01),
        arrowprops=arrowprops
    )

    
    # Level lines
    x1,x2 = potA_inv(E_A)
    ax.plot([x1+0.01,x2-0.01], [E_A, E_A], color = 'tab:blue')
    ax.plot([x1+0.01,x2-0.01], [E_Ap, E_Ap], color = 'tab:blue')
    
    # Harmonic oscillators
    x = np.linspace(x_A-0.44, x_A+0.44, 400)
    ax.plot(x, potA(x), color = 'black')
    ax.plot(x, potAp(x), color = 'black')
    
    # Labels
    ax.text(potA_inv(E_A)[0]-0.65, E_A-0.05, r"$\mathrm{A}^-$")
    ax.text(potA_inv(E_A)[1]+0.1, E_A-0.05, r"$\nu_{\mathrm{A}^-}$")
    
    ax.text(potA_inv(E_A)[0]-0.65, E_Ap-0.05, r"$\mathrm{A} + e_k^-$")
    ax.text(potA_inv(E_A)[1]+0.1, E_Ap-0.05, r"$\nu_\mathrm{A}$")
    
    # Level lines
    x1,x2= potD_inv(E_D)
    ax.plot([x1+0.01,x2-0.01], [E_D, E_D], color = 'tab:blue')
    ax.plot([x1+0.01,x2-0.01], [E_Dp, E_Dp], color = 'tab:blue')
    
    # Harmonic oscillators
    x = np.linspace(x_D-0.44, x_D+0.44, 400)
    ax.plot(x, potD(x), color = 'black')
    ax.plot(x, potDp(x), color = 'black')

    # Arrow
    ax.annotate(
        "",
        xy=(x_D, shift_D+E_D+0.01),
        xytext=(x_D, E_D-0.025),
        arrowprops=arrowprops
    )

    # Labels
    ax.text(potD_inv(E_D)[1]+0.2, E_D-0.05, r"$\mathrm{D}$")
    ax.text(potD_inv(E_D)[0]-0.3, E_D-0.05, r"$\nu_{\mathrm{D}}$")
    
    ax.text(potD_inv(E_D)[1]+0.2, E_Dp-0.05, r"$\mathrm{D}^+ + e_{k\prime}^-$")
    ax.text(potD_inv(E_D)[0]-0.3, E_Dp-0.05, r"$\nu_{\mathrm{D}^+}$")
    
    # omega
    # https://stackoverflow.com/questions/33707162/zigzag-or-wavy-lines-in-matplotlib
    ax.text((x_A+x_D)/2 - 0.075, E_A+shift_A/2 + 0.3, r"$\omega$", color='tab:red')
    rcParams['path.sketch'] = (4, 15, 1)
    ax.plot([x_A+0.005, x_D-0.005], [E_A+shift_A/2, E_D+shift_D/2], lw=1.5, color='tab:red')
    cfg_plot.set_rcParams()

    ax.axis("off")
    fname = os.path.join(DIR_PLOTS, "icec_energy_sketch.pdf")
    fig.savefig(fname, bbox_inches='tight')