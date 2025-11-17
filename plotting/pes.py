import mpmath
import matplotlib.pyplot as plt
from icec.intraIcec import IntraICEC
from icec.morse import Morse
from icec.constants import Units
from plotting.base import DIR

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 16})

def plot_vib_state(ax, morse:Morse, vi, scale=1./15, yshift=0):
    psi = [morse.psi(vi,r_i) * scale
               + morse.energy(vi)*Units.HARTREE2EV + yshift 
               for r_i in morse.r]
    ax.plot(morse.r*Units.BOHR2ANGSTROM, psi, color='tab:blue', lw=1)
    
def plot_diss_state(ax, morse:Morse, energy, norm=None, scale=1, yshift=0, color='lightskyblue'):
    if norm is None:
        norm = morse.norm_diss(energy)
    psi_diss = [mpmath.re(norm * morse.psi_diss(energy,r_i)) * scale
                + energy*Units.HARTREE2EV + yshift 
                for r_i in morse.r]
    ax.plot(morse.r*Units.BOHR2ANGSTROM, psi_diss, color=color, lw=1)

def plot_PES(icec:IntraICEC, system, L=5*Units.ANGSTROM2BOHR, yshift=0):
    # TODO I defined bound states to have negative energies, recheck the y values
    # TODO annotations as inputs
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,  height_ratios=[0.3, 0.7], figsize=(5,5))
    fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
    ax2.set_xlabel(r'$R$ [$\mathrm{\AA}$]')
    
    ax2.set_ylim(-icec.Morse_D.De*Units.HARTREE2EV-0.1, 0.1)
    height = 0.3/0.7*(0.1 - (-icec.Morse_D.De*Units.HARTREE2EV-0.1))
    ax1.set_ylim(yshift-icec.Morse_Dp.De*Units.HARTREE2EV-0.1, yshift-icec.Morse_Dp.De*Units.HARTREE2EV-0.1 + height)
    
    r = icec.Morse_D.make_rgrid(rmax=L)
    icec.Morse_Dp.r = r
    scale = 1./15
    # D
    for vi in range(3):
        plot_vib_state(ax2, icec.Morse_D, vi, scale)
        
    V = icec.Morse_D.V(r)
    ax2.plot(r*Units.BOHR2ANGSTROM, V*Units.HARTREE2EV, color='black')
    ax2.annotate(r'$\mathrm{LiH}$', (r[-100]*Units.BOHR2ANGSTROM, V[-100]*Units.HARTREE2EV - 0.25))
    
    # Dp
    energy, norm = icec.Morse_Dp.diss_energies[30], icec.Morse_Dp.diss_norms[30]
    plot_diss_state(ax1, icec.Morse_Dp, energy, norm, scale, yshift)
    energy, norm = icec.Morse_Dp.diss_energies[0], icec.Morse_Dp.diss_norms[0]
    plot_diss_state(ax1, icec.Morse_Dp, energy, norm, scale, yshift)
    
    plot_vib_state(ax1, icec.Morse_Dp, icec.Morse_Dp.vmax, scale, yshift)
    plot_vib_state(ax1, icec.Morse_Dp, 0, scale, yshift)
    
    V = icec.Morse_Dp.V(r)
    ax1.plot(r*Units.BOHR2ANGSTROM, V*Units.HARTREE2EV + yshift, color='black')
    ax1.annotate(r'$\mathrm{LiH}^+$', (r[-100]*Units.BOHR2ANGSTROM, V[-100]*Units.HARTREE2EV + yshift + 0.1))
    
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(bottom=False)
    
    # cut out slanted lines
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    
    fig.text(0, 0.5, r'$E$ [eV]', va='center', rotation='vertical')
    fname = DIR + f"plots/{system}.PES.L{round(L*Units.BOHR2ANGSTROM)}.pdf"
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.2)
