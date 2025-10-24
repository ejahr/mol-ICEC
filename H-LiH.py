import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from input.HLiH import LiH, LiHp, Hp_LiH, Bp_LiH
from icec.icec import ICEC
from icec.intraIcec import IntraICEC
from icec.morse import Morse
from icec.constants import Units, Constants

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 16})
width, height = 6, 4

DIR = '/home/elena/intraICEC/dimers/'

#https://doi.org/10.1021/jp9921295
R = 2 * Units.ANGSTROM2BOHR
#R = 10 * Units.ANGSTROM2BOHR
R = np.array([6,8,10]) * Units.ANGSTROM2BOHR

min_kinE = 0.01 * Units.EV2HARTREE
max_kinE = 9 * Units.EV2HARTREE
resolution = 1000

def plot_xs_vB_vBp(system, icec: IntraICEC, R):
    fname = DIR + "plots/" + system + '.all_vib.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.pdf"
    with PdfPages(fname) as pdf:
        energies = icec.energyGrid*Units.HARTREE2EV
        for vi in range(LiH.v_max+1): 
            fig, ax = plt.subplots()
            ax.set_yscale('log')
            ax.set_xlabel(r'$\epsilon$ [eV]')
            ax.set_ylabel(r'$\sigma$ [Mb]')
            ax.set_xlim(-0.2, 8.5)
            ax.set_ylim(1e-5, 1e2)
            ax.grid(True)
            results = read_results_file(system, R)
            ax.plot(results[:,0], results[:, vi+1], label='total', color='grey')
            for vf in range(LiHp.v_max+1):
                label = r'$v_{LiH^+}=$' + str(vf)
                xs = icec.xs_vD_vDp(R, vi, vf)
                ax.plot(energies, xs, label=label)
            icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', ls=':', zorder=0)
            ax.legend()
            pdf.savefig(fig)  #, bbox_inches = "tight"
            plt.close(fig) 
    
def calculate_xs_bb(system, header, icec: IntraICEC, R, v_max, vp_max, modifier=''):
    xs_array = icec.energyGrid*Units.HARTREE2EV
    for v in range(v_max+1):
        xs = icec.xs_vD(R, v, vp_max)
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
    file_path = DIR + "results/" + system + '.xs' + modifier + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=header)
    
def calculate_xs_R(system, icec, R, header):
    for r in R:
        headerR = header + f'R = {round(r*Units.BOHR2ANGSTROM)} Angstrom'
        headerR += 'E_in [eV] | xs [Mb]'
        calculate_xs_bb(system, headerR, icec, r, LiH.v_max, LiHp.v_max)

def calculate_spectrum(system, header, icec: IntraICEC, R, electronE, modifier=''): 
    new_header = header + "E_in = " + str(round(electronE*Units.HARTREE2EV)) + " eV\n"
    new_header += "| E_out [eV] : xs [Mb] |"  
    spectrum_all_vi = np.array([]) 
    for vi in range(LiH.v_max+1):
        spectrum = icec.spectrum(electronE, R, vi, LiHp.v_max)
        if spectrum_all_vi.size == 0:
            spectrum_all_vi = spectrum
        else:
            spectrum_all_vi = np.hstack((spectrum_all_vi, spectrum))          
    fname = DIR + "results/" + system + ".spectrum" + modifier + ".E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    np.savetxt(fname, spectrum_all_vi, fmt='%1.3e', header=new_header)  
    
def read_results_file(system, R, modifier=''):
    file_path = DIR + "results/" + system + '.xs' + modifier + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    results = np.loadtxt(file_path, comments='#')
    return results

def plot_xs(ax, system, R, v_B, label='icec', modifier='', **kwargs):
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    ax.set_xlim(-0.2, 8.6)
    # ax.plot(icec.energyGrid * Units.HARTREE2EV,  icec.PI_xs_B(v_B, 0, icec.energyGrid + icec.IP_A)*Units.AU2MB, label=r'$\sigma_\text{PI}$')
    results = read_results_file(system, R, modifier=modifier)
    ax.plot(results[:,0], results[:, v_B+1], label=label, **kwargs)
    
def plot_xs_FC(system, icec: IntraICEC, R, icec_fixed:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    if icec_fixed is not None:
        energy = icec_fixed.energyGrid*Units.HARTREE2EV
        xs = icec_fixed.xs_energy(R)
        ax.plot(energy, xs, color='gray', label=r'unresolved')
    vi = 0
    plot_xs(ax, system, R, vi, label=r'FC', modifier='-FC', color='tab:red')
    plot_xs(ax, system, R, vi, label=r'resolved', color='tab:blue')
    fname = DIR + 'plots/' + system + '.xs-FC.v0.R'+ str(round(R*Units.BOHR2ANGSTROM)) + '.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_xs_vi(system, icec: IntraICEC, R, icec_fixed:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    if icec_fixed is not None:
        energy = icec_fixed.energyGrid*Units.HARTREE2EV
        xs = icec_fixed.xs_energy(R)
        ax.plot(energy, xs, color='gray', label='unresolved')
        
    color = ['tab:red', 'tab:purple', 'tab:blue']
    for vi in range(0, LiH.v_max+1):
        label = r'$v_i=$' + str(vi)
        plot_xs(ax, system, R, vi, label, color=color[vi])
        
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', ls=':', zorder=0)
    
    ax.legend()
    fname = DIR + 'plots/' + system + '.vB.R'+ str(round(R*Units.BOHR2ANGSTROM)) + '.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_xs_vi_FC(system, icec: IntraICEC, R, icec_fixed:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    
    if icec_fixed is not None:
        energy = icec_fixed.energyGrid*Units.HARTREE2EV
        xs = icec_fixed.xs_energy(R)
        ax.plot(energy, xs, color='gray', label='unresolved')
        
    color = ['tab:red', 'tab:purple', 'tab:blue']
    for vi in range(0, LiH.v_max+1):
        label = r'$v_i=$' + str(vi)
        plot_xs(ax, system, R, vi, label, color=color[vi])
        plot_xs(ax, system, R, vi, label+' FC', modifier='-FC', linestyle='--', color=color[vi])
    
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', ls=':', zorder=0)    
    
    fname = DIR + 'plots/' + system + '.xs-FC.vB.R'+ str(round(R*Units.BOHR2ANGSTROM)) + '.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_xs_boltzmann(system, icec: IntraICEC, R, T, vib_energies=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.set_yscale('log')
    ax.set_xlim(-0.2, 8.5)
    ax.grid(True)
    results = read_results_file(system, R)
    
    blues = plt.get_cmap("Blues_r")    
    for t in T:
        if vib_energies is None:
            # add De to energy(vi) to get positive values which increases numerical stability
            norm = sum(np.exp(-(icec.Morse_D.energy(vi)+icec.Morse_D.De)/Constants.KB/t) 
                       for vi in range(LiH.v_max+1)
                       )
            avg = sum(
                np.exp(-(icec.Morse_D.energy(vi)+icec.Morse_D.De)/Constants.KB/t) * results[:, vi+1] 
                for vi in range(LiH.v_max+1)
                )
        else:
            norm = sum(
                np.exp(-vib_energies[vi]/Constants.KB/t) 
                for vi in range(LiH.v_max+1)
                )
            avg = sum(
                np.exp(-vib_energies[vi]/Constants.KB/t) * results[:, vi+1]
                for vi in range(LiH.v_max+1)
                )
        label = r'$T=$' + str(t) + 'K'
        blue = blues(T.index(t) / (len(T) + 1 / len(T)))
        ax.plot(results[:,0], avg/norm, label=label, color=blue)
        
    #for vi in range(v_max + 1):
    #    plot_xs(ax, system, R, vi, r'$v_{LiH}=$'+str(vi), linestyle=':')  
        
    ax.legend()
    plt.tight_layout()
    fname = DIR + 'plots/' + system + '.boltzmann.R'+ str(round(R*Units.BOHR2ANGSTROM)) + '.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    

def plot_xs_R(system, icec: IntraICEC, R, icec_fixed:ICEC=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca() 

    blues = plt.get_cmap("Blues_r")
    for r in R:
        blue = blues(R.index(r) / (len(R) + 1 / len(R)))
        label = r'$R=$' + str(round(r*Units.BOHR2ANGSTROM)) + r'$\,\mathrm{\AA}$'
        
        if icec_fixed is not None:
            energy = icec_fixed.energyGrid*Units.HARTREE2EV
            xs = icec_fixed.xs_energy(r)
            ax.plot(energy, xs, color=blue, ls='--')
                
        plot_xs(ax, system, r, 0, label, color=blue)
    
    icec.plot_PR_xs(ax, label=r'$\sigma_\text{PR}$', color='gray', ls=':', zorder=0)
    plt.legend()
    fname = DIR + 'plots/' + system + '.R.icec.pdf'
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_spectrum(system, R, electronE, vi=0, title=None, icec_fixed:ICEC=None, modifier=''):
    fname = DIR + "results/" + system + ".spectrum.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    results = np.loadtxt(fname, comments='#')
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon_\text{out}$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.grid(True)
    
    ax.hlines(icec.PR_xs_A(electronE)*Units.AU2MB, 0, 10, color='gray', ls=':', zorder=0)   
    
    color = ['tab:red', 'tab:purple', 'tab:blue']

    bars = [None] * (LiH.v_max+1)
    for vi in range(LiH.v_max+1):
        label = r'$v_i=$' + str(vi)
        bars[vi] = ax.bar(results[:,3*vi], results[:,3*vi+1], width=0.002, label=label, color=color[vi])
    labels = [r'$v_i=$' + str(vi) for vi in range(LiH.v_max+1)] 
        
    if icec_fixed is not None:
        energy_out = icec_fixed.electronE_f(electronE)
        xs = icec_fixed.xs(electronE, R)
        bars.append(ax.bar(energy_out*Units.HARTREE2EV, xs*Units.AU2MB, width=0.002, color='black', label='unresolved'))
    labels.append('unresolved')
        
    ax.legend(handles=[bar[0] for bar in bars], labels=labels, ncols=2, fontsize='small', loc='upper right')
    
    x_min = min(rect.get_x() for bar in bars for rect in bar)
    x_max = max(rect.get_x() for bar in bars for rect in bar)
    ax.set_xlim(x_min - 0.025, x_max + 0.025)
    
    ax.annotate(r'$\sigma_\text{PR}$', (x_max + 0.025, icec.PR_xs_A(electronE)*Units.AU2MB), xytext=(3,-3),    # fraction, fraction
            textcoords='offset points', color='gray')
    
    fname = DIR + 'plots/' + system + ".spectrum" + modifier + ".E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.pdf"
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_spectrum_FC(system, R, electronE, vi=0, title=None):
    fname = DIR + "results/" + system + ".spectrum-FC.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    results_FC = np.loadtxt(fname, comments='#')
    
    fname = DIR + "results/" + system + ".spectrum.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.txt"
    results_resolved = np.loadtxt(fname, comments='#')
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca() 
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon_\text{out}$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    ax.set_ylim(2*1e-4, 0.2)
    ax.grid(True)
    
    color_resolved = ['tab:red', 'tab:purple', 'tab:blue']
    color_FC = ['tab:orange', 'violet' ,'lightskyblue']
    for vi in range(LiH.v_max+1):
        label = r'$v_i=$' + str(vi)
        ax.bar(results_resolved[:,3*vi], results_resolved[:,3*vi+1], width=0.004, color=color_resolved[vi], label=label)
        ax.bar(results_FC[:,3*vi], results_FC[:,3*vi+1], width=0.002, color=color_FC[vi], label=label + ' FC')

    ax.legend(ncols=3, fontsize='small', loc='upper center')
    fname = DIR + 'plots/' + system + ".spectrum-FC.E"+ str(round(electronE*Units.HARTREE2EV)) + '.R'+ str(round(R*Units.BOHR2ANGSTROM)) + ".icec.pdf"
    plt.tight_layout()
    fig.savefig(fname)
    
def plot_morse(icec:IntraICEC, system):
    # TODO I defined bound states to have negative energies, recheck the y values
    yshift = (8.066308039 - 7.781734076) * Units.HARTREE2EV 
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,  height_ratios=[0.4, 0.6], figsize=(5,5))
    fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
    ax2.set_xlabel(r'$R$ [$\mathrm{\AA}$]')
    
    ax2.set_ylim(-0.1, 2.5)
    ax1.set_ylim(yshift-0.1, yshift - 0.1 + 0.4/0.6*(2.5+0.1))
    
    r = icec.Morse_D.make_rgrid(rmax=5*Units.ANGSTROM2BOHR)
    V = icec.Morse_D.V(r)
    ax2.plot(r*Units.BOHR2ANGSTROM, V*Units.HARTREE2EV, color='black', label=r'$\mathrm{LiH}$')
    ax2.annotate(r'$\mathrm{LiH}$', (r[-100]*Units.BOHR2ANGSTROM, V[-100]*Units.HARTREE2EV - 0.25))
    
    for vi in range(LiH.v_max+1):
        psi = [icec.Morse_D.psi(vi,r_i)/15 + icec.Morse_D.energy(vi)*Units.HARTREE2EV for r_i in r]
        ax2.plot(r*Units.BOHR2ANGSTROM, psi, color='black', lw=1)
    
    V = icec.Morse_Dp.V(r)
    ax1.plot(r*Units.BOHR2ANGSTROM, V*Units.HARTREE2EV + yshift, color='black', ls='--', label=r'$\mathrm{LiH}^+$')
    ax1.annotate(r'$\mathrm{LiH}^+$', (r[-100]*Units.BOHR2ANGSTROM, V[-100]*Units.HARTREE2EV + yshift + 0.05))
    
    psi = [icec.Morse_Dp.psi(0,r_i)/15 + yshift + icec.Morse_Dp.energy(0)*Units.HARTREE2EV for r_i in r]
    ax1.plot(r*Units.BOHR2ANGSTROM, psi, color='black', lw=1)
    
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(bottom=False)
    
    # cut out slanted lines
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    
    fig.text(0.04, 0.5, r'$E$ [eV]', va='center', rotation='vertical')
    
    fname = DIR + 'plots/' + system + ".PES.pdf"
    fig.savefig(fname)

def test_FC_factors(icec_fixed: ICEC, icec:IntraICEC, icec_FC:IntraICEC):
    omega = 10*Units.EV2HARTREE
    print('xs         ', icec.PI_xs_D(0,0,omega))
    print('xs FC      ', icec_FC.PI_xs_D(0,0,omega))
    print('xs FC paper', icec_fixed.PI_xs_B(omega*Units.HARTREE2EV)*Units.MB2AU*0.0153)
    
    for vp in range(0, LiH.v_max+1):
        icec_FC.FC_factor(0,vp)
        
def plot_H_PI_PR(icec:ICEC):
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_title('Hydrogen')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ [eV]')
    ax.set_ylabel(r'$\sigma$ [Mb]')
    
    PI_xs = np.array([])
    hbaromega = np.array([])
    for electronE in icec.energyGrid:
        omega = electronE + icec.IP_A
        hbaromega = np.append(hbaromega, [omega*Units.HARTREE2EV])
        xs = icec.PI_xs_A(omega*Units.HARTREE2EV)
        PI_xs = np.append(PI_xs, [xs])
    ax.plot(icec.energyGrid*Units.HARTREE2EV, PI_xs, label = r'$H\to H^+$')
    icec.plot_PR_xs(ax, label = r'$H^+\to H$')
    ax.legend()
    fname = DIR + 'plots/H.PI.PR.pdf'
    fig.savefig(fname)
    
def test_roots(Morse:Morse):
    fname = DIR + 'data/LiH/LiHp.diss_energies.L' + str(int(Morse.box_length*Units.BOHR2ANGSTROM)) + 'A.txt'
    root_estimates, roots = Morse.find_roots(fname)
    fig = plt.figure()
    ax = plt.gca() 
    ax.set_title('Dissociative states')
    ax.set_xlabel(r'$E$ [eV]')
    ax.set_ylabel(r'$E$ [a.u.]')
    ax.set_yscale('log')
    ax.bar(root_estimates*Units.HARTREE2EV, root_estimates/2, width=0.005, color='tab:blue', label='estimates')
    ax.bar(roots*Units.HARTREE2EV, roots, width=0.005, color='tab:red', label='roots')
    ax.legend()
    fname = DIR + 'plots/LiHp_roots.pdf'
    fig.savefig(fname)

HLi = True
BLi = False
calculation = True

electronE = 1*Units.EV2HARTREE
R = 6*Units.ANGSTROM2BOHR
R_list = [6,8,10]
R_list = [r*Units.ANGSTROM2BOHR for r in R_list]
  
if HLi:
    system = 'Hp-LiH'
    title = r'$\text{H}^+ \text{LiH}$'
    
    icec_fixed = ICEC(*Hp_LiH.input_fixed)
    
    icec_fixed.make_energy_grid(min_kinE*Units.HARTREE2EV, LiH.max_kinE_unresolved*Units.HARTREE2EV, resolution)
    
    icec = IntraICEC(*Hp_LiH.input)
    icec.input_vib_spacing_D(LiH.vib_spacing, LiHp.vib_spacing)
    icec.make_energy_grid(min_kinE, max_kinE, resolution)
    icec.define_Morse_D(*LiH.morse_parameters, wexe=LiH.wexe)
    icec.define_Morse_Dp(*LiHp.morse_parameters, wexe=LiHp.wexe)
    icec.define_PI_xs_D(method="resolved")
    
    icec.Morse_Dp.define_box(10*Units.ANGSTROM2BOHR)
    test_roots(icec.Morse_Dp)
    
    icec_FC = IntraICEC(*Hp_LiH.input_unresolved)
    icec_FC.define_Morse_D(*LiH.morse_parameters)
    icec_FC.define_Morse_Dp(*LiHp.morse_parameters)
    icec_FC.make_energy_grid(min_kinE, max_kinE, resolution)
    icec_FC.define_PI_xs_D(method="FC")
    
    test_FC_factors(icec_fixed, icec, icec_FC)
    
    plot_morse(icec, 'LiH')
    
    system = 'Hp-LiH'
    header = 'e- + H+ + LiH -> H + LiH+ + e-\n'
    header += f'Number of initial vibrational states: {LiH.v_max+1}\n' 
    header += f'Number of final vibrational states: {LiHp.v_max+1}\n' 

    if calculation:
        calculate_xs_bb(system, header, icec, R, LiH.v_max, LiHp.v_max)
        calculate_xs_bb(system, header, icec_FC, R, LiH.v_max, LiHp.v_max, modifier='-FC')
        calculate_xs_R(system, icec, R_list, header)
    
    plot_xs_vi(system, icec, R, icec_fixed=icec_fixed)
    plot_xs_FC(system, icec, R, icec_fixed=icec_fixed)
    plot_xs_R(system, icec, R_list, icec_fixed=icec_fixed)

    if calculation:
        calculate_spectrum(system, header, icec, R, electronE)
        calculate_spectrum(system, header, icec_FC, R, electronE, modifier='-FC')

    plot_spectrum(system, R, 1*Units.EV2HARTREE, icec_fixed=icec_fixed)
    plot_spectrum_FC(system, R, 1*Units.EV2HARTREE)

    T = [15, 298, 2000] 
    plot_xs_boltzmann(system, icec, R, T, LiH.vib_energies)
    plot_xs_vB_vBp(system, icec, 4*Units.ANGSTROM2BOHR)

if BLi:
    system = 'Bp-LiH'
    title = r'$\text{B}^+ \text{LiH}$'
    
    icec = IntraICEC(*Bp_LiH.input)  
    #icec.input_vib_spacing_D(vib_spacing_LiH, vib_spacing_LiHp)
    icec.define_Morse_D(*LiH.morse_parameters)
    icec.define_Morse_Dp(*LiHp.morse_parameters)
    icec.make_energy_grid(min_kinE, max_kinE, resolution) 
    
    R = 10 * Units.ANGSTROM2BOHR
    
    header = 'e- + B+ + LiH -> B + LiH+ + e-\n'
    header += f'Number of initial vibrational states: {LiH.v_max+1}\n' 
    header += f'Number of final vibrational states: {LiHp.v_max+1}\n' 
    header += f'R = {round(R*Units.BOHR2ANGSTROM)} Angstrom'
    header += 'E_in [eV] | xs [Mb]'
        
    #xs_bb(system, header, icec, R, v_max, vp_max)
    
    T = [15, 298, 2000] 
    plot_xs_boltzmann(system, icec, R, T, LiH.vib_energies)
    plot_xs_vi(system, icec, R, title=r"\mathrm{B}^+ + \mathrm{LiH}")

    #for electronE in electron_energies:
    #    calculate_spectrum(system, header, icec, R, electronE)

    #plot_spectrum(system, R, 1*Units.EV2HARTREE, title=title)
    #plot_spectrum(system, R, 5*Units.EV2HARTREE, title=title)