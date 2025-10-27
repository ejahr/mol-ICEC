import numpy as np
import scipy as sp
import mpmath
from typing import Callable
from .constants import Constants, Units
from .morse import Morse

class IntraICEC:
    """ICEC cross section with diatomic molecules.

    Uses atomic units (hbar=1, me=1, hartree energy=1).

    Args:
        degeneracyFactor(float) : degeneracy of A- divided by degeneracy of A. g_{A^-} / g_A
        IP (float): Ionization potential (Hartree)
        PI_xs_A (float -> float): Fit for Photoionization cross section of A (a.u. -> a.u.)
        file_PI_xs_D (str) : file name of the photionization cross section of D

    Attributes:
        degeneracyFactor(float) : degeneracy of A- divided by degeneracy of A. g_{A^-} / g_A
        IP (float): Ionization potential (Hartree)
        PI_xs_A (float -> float): Fit for Photoionization cross section of A (a.u. -> a.u.)
        file_PI_xs_D (str) : file name of the photionization cross section of D
        prefactor (float): collects terms that are neither energy nor R dependent 
    """    
    def __init__(self, degeneracyFactor: float, IP_A: float, IP_D: float, PI_xs_A: Callable, file_PI_xs_D: str) :
        self.degeneracyFactor = degeneracyFactor # g_A / g_A+
        self.IP_A = IP_A 
        self.IP_D = IP_D # assumption: adiabatic ionization energy
        self.PI_xs_A = PI_xs_A
        self.file_PI_xs_D = file_PI_xs_D
        self.prefactor = (3 * Constants.c**2) / (8 * np.pi)
            
    def define_Morse_D(self, mu:float, we:float, re:float, De:float, wexe:float=0):
        """Morse potential for the initial vibrational mode of the system.
        - mu: reduced mass (proton mass)
        - we: Morse parameter (a.u.)
        - re: Equilibrium bond distance (a.u.)
        - De: Dissociation energy (a.u.)
        """
        self.Morse_D:Morse = Morse(mu, we, re, De, wexe)

    def define_Morse_Dp(self, mu:float, we:float, re:float, De:float, wexe:float=0):
        """Morse potential for the initial vibrational mode of the system.
        - mu: reduced mass (proton mass)
        - we: Morse parameter (a.u.)
        - re: Equilibrium bond distance (a.u.)
        - De: Dissociation energy (a.u.)
        """
        self.Morse_Dp:Morse = Morse(mu, we, re, De, wexe)

    def make_energy_grid(self, minEnergy=0.01*Units.EV2HARTREE, maxEnergy=10*Units.EV2HARTREE, resolution=100, geometric=True): 
        """ Make a suitable grid of incoming electron energies.
        - Energy (a.u.)
        - resolution : number of grid points
        """
        if geometric:
            self.energyGrid = np.geomspace(minEnergy, maxEnergy, resolution)
        else:
            self.energyGrid = np.linspace(minEnergy, maxEnergy, resolution)

    def make_R_grid(self, Rmin=2*Units.ANGSTROM2BOHR, Rmax=10*Units.ANGSTROM2BOHR, resolution=100): 
        """ Make a suitable grid of interatomic distances.
        - R (Bohr, a.u.)
        - resolution : number of grid points
        """
        self.rGrid = np.linspace(Rmin, Rmax, resolution)
        
    def define_PI_xs_D(self, method="FC"):
        if method == 'FC':
            self.PI_xs_D = self.PI_xs_D_FC
        elif method == 'resolved':
            self.PI_xs_D = self.PI_xs_D_resolved
        else:
            raise ValueError(
                f"Invalid method '{method}'. Valid options are: 'FC', 'resolved'."
            )
            
    # TODO keep interp_function in memory
    def PI_xs_D_resolved(self, vD:int, vDp:int, hbarOmega:float):
        filename = self.file_PI_xs_D + f"{vD}_{vDp}.txt"
        data = np.loadtxt(filename)
        energies, xs = data[:, 0]*Units.EV2HARTREE, data[:, 1]*Units.MB2AU
        if hbarOmega >= energies[-1]:
            return np.nan
        interp_func = sp.interpolate.interp1d(
            energies, xs, kind='linear'
            )
        return interp_func(hbarOmega)
    
    def FC_factor(self, vD:int, vDp:int) -> float:
        """Returns the Franck-Condon factor <psi_vi|psi_vf> corresponding to the photoionization.
        
        Args:
            vD (int): vibrational quantum number of D.
            vDp (int): vibrational quantum number of D+.

        Returns:
            float: Franck-Condon factor.
        """
        if not hasattr(self, 'FC_factor_saved'):
            # vmax+1 results in out of bound... temp fix: +10 
            self.FC_factor_saved = np.zeros((self.Morse_D.vmax+10, self.Morse_Dp.vmax+10))
        if self.FC_factor_saved[vD][vDp] > 0:
            return self.FC_factor_saved[vD][vDp]
        else:
            def integrand(r):
                return mpmath.conj(self.Morse_D.psi(vD, r)) * self.Morse_Dp.psi(vDp, r)
        #result, error = sp.integrate.quad(integrand, 0, np.inf)
            result = mpmath.quad(integrand, [0, 5*Units.ANGSTROM2BOHR], maxdegree=10)
            self.FC_factor_saved[vD][vDp] = np.abs(result)**2
            return np.abs(result)**2
        
    def PI_xs_D_electronic(self, hbarOmega:float) -> float:
        data = np.loadtxt(self.file_PI_xs_D + '.txt')
        energies, xs = data[:, 0]*Units.EV2HARTREE, data[:, 1]*Units.MB2AU
        if hbarOmega >= energies[-1]:
            return np.nan
        interp_func = sp.interpolate.interp1d(
            energies, xs, kind='linear', fill_value=np.nan
            )
        return interp_func(hbarOmega) 
    
    def PI_xs_D_FC(self, vD:int, vDp:int, hbarOmega:float) -> float:
        return self.PI_xs_D_electronic(hbarOmega) * self.FC_factor(vD, vDp)
    
    def hbarOmega(self, electronE:float) -> float:
        return electronE + self.IP_A 
    
    def electronE_f(self, electronE:float, vD:int, vDp:int) -> float:
        if vDp is None:
            vib_energy_D = 0
        elif hasattr(self, 'vib_diff_to_v0_D'):
            vib_energy_D = self.vib_diff_to_v0_Dp[vDp] - self.vib_diff_to_v0_D[vD]
        else:
            vib_energy_D = (self.Morse_Dp.energy(vDp) - self.Morse_Dp.energy(0)) - (self.Morse_D.energy(vD) - self.Morse_D.energy(0))
        return self.hbarOmega(electronE) - (self.IP_D + vib_energy_D)
    
    def input_vib_spacing_D(self, vib_spacing_D, vib_spacing_Dp):
        self.vib_diff_to_v0_D = np.cumsum(vib_spacing_D)
        self.vib_diff_to_v0_Dp = np.cumsum(vib_spacing_Dp)

    # ====== CROSS SECTION ======
       
    def xs(self, electronE:float, R:float, vD:int=0, vDp:int=0) -> float:
        """ Calculates the ICEC cross section (a.u.) for some kinetic energy and R.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - R: internuclear distance: (Bohr, a.u.)
        - v_A+ -> v_A (v_A -> v_A+ Photoionization)
        - v_D -> v_D+
        """   
        if self.electronE_f(electronE, vD, vDp) <= 0: 
            return 0
        else: 
            # TODO
            hbarOmega = self.hbarOmega(electronE)
            PI_xs_A = self.PI_xs_A(hbarOmega)
            PI_xs_D = self.PI_xs_D(vD, vDp, hbarOmega)
            return self.prefactor * self.degeneracyFactor * PI_xs_A * PI_xs_D / (electronE * hbarOmega**2 * R**6)

    def xs_vD_vDp(self, R:float, vD:int=0, vDp:int=0):
        """ Calculates the ICEC cross section (Mb) for given range of kinetic energies.
        - R: internuclear distance: (Bohr, a.u.)
        """        
        xs = np.array([
            self.xs(energy, R, vD, vDp)
            for energy in self.energyGrid
        ]) 
        return xs * Units.AU2MB
    
    def xs_vD(self, R:float, vD:int, vDp_max:int):
        """ Cross section [Mb] for vi -> bound states over range of electron energies.
        """
        # Element-wise summation sum(list_of_arrays)
        xs = sum(
            self.xs_vD_vDp(R, vD, vDp) 
            for vDp in range(vDp_max + 1)
        )
        return xs
    
    def xs_boltzmann(self, R:float, t:float, vD_max:int, vDp_max:int):
        # add De to energy(vi) to get positive values, increasing numerical stability
        norm = sum(
            np.exp(-(self.Morse_D.energy(vD)+self.Morse_D.De)/Constants.KB/t) 
            for vD in range(vD_max+1)
        )
        avg = sum(
            np.exp(-(self.Morse_D.energy(vD)+self.Morse_D.De)/Constants.KB/t) * self.xs_vD(R, vD, vDp_max)
            for vD in range(vD_max+1)
        )
        return avg/norm * Units.AU2MB
    
    def spectrum(self, electronE:float, R:float, vD:int=0, vDp_max:int=0):
        """ Cross sections [Mb] for vi -> bound states given some electron energy.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        """
        spectrum = []
        for vDp in range(vDp_max+1):
            electronE_f = self.electronE_f(electronE, vD, vDp)
            if electronE_f >= 0:
                xs = self.xs(electronE, R, vD, vDp)
                spectrum.append([electronE_f * Units.HARTREE2EV, xs * Units.AU2MB, vDp])
        return np.array(spectrum)

    def xs_R(self, electronE:float, vD:int=0, vDp:int=None):
        """ Calculates ICEC cross section (Mb) for given range of interatomic distances.
        - electronE : energy of incoming electron (Hartree, a.u.) 
        - R : interatomic distance (Bohr, a.u.)
        """
        if not hasattr(self, 'rGrid'):
            self.make_R_grid()
        xs = np.array([
            self.xs(electronE, r, vD, vDp)
            for r in self.rGrid
        ])
        return xs * Units.AU2MB
    
    # ====== DISSOCIATION OF D ======
    
    def integrate_r(self, integrand, vi, E, lower_bound=None):
        '''Integration over r
        - integrand : function to be integrated
        - electronE : kinetic energy of incoming electron [Hartree, a.u.]
        - vi: initial vibrational quantum number
        - E : energy of the dissociative Morse state [Hartree]
        '''
        if lower_bound is None:
            lower_bound = self.Morse_Dp.get_lower_bound(E)
        r_left = min(self.Morse_D.reflection_point_left(self.Morse_D.energy(vi)), self.Morse_Dp.reflection_point_left(E))
        if r_left <= lower_bound:
            raise Exception("r_left <= lower_bound\n")
        r_right = max(self.Morse_D.rmax, self.Morse_Dp.rmax)

        num_oscillation = self.Morse_Dp.estimate_oscillation(E)
        num_intervals = (vi + 1) * num_oscillation
        if num_intervals < 10:
            return mpmath.quadsubdiv(integrand, [lower_bound, r_left, r_right, self.Morse_Dp.box_length], maxdegree=10)
        else:
            result = mpmath.quadsubdiv(integrand, [lower_bound, r_left], maxdegree=10)
            intervals_mid = np.linspace(r_left, r_right, num_intervals)
            result += mpmath.quadsubdiv(integrand, intervals_mid, maxdegree=10)
            intervals_high = np.linspace(r_left, self.Morse_Dp.box_length, num_oscillation+1)
            result += mpmath.quadsubdiv(integrand, intervals_high, maxdegree=10)
            return result
        
    def FC_bc_integrand(self, vi, E, r):
        return mpmath.conj(self.Morse_Dp.psi_diss(E, r)) * self.Morse_D.psi(vi, r)
    
    def FC_bc(self, vi:int, E:float, lower_bound:float=None, norm:float=None):
        '''Franck-Condon (FC) factor for a bound to continuum (bc) transition |<psi_E|psi_v>|^2
        norm: normalization constant for the vibrational continuum state
        divide integration into intervals to deal with highly oscillating integrand
        '''
        if norm is None:
            norm = self.Morse_Dp.norm_diss(E)
        def integrand(r):
            return mpmath.conj(self.Morse_Dp.psi_diss(E, r)) * self.Morse_D.psi(vi, r)
        result = self.integrate_r(integrand, vi, E, lower_bound=lower_bound)    
        return (mpmath.fabs(norm * result)) ** 2
    
    def electronE_f_bc(self, electronE:float, vD:int, E:float) -> float:
        """Kinetic energy of the outgoing electron
        - electronE: kinetic energy of the incoming electron (Hartree)
        - vD: vibrational quantum number of the initial state
        - E: energy of the dissociative final state, i.e. energy above dissociation limit (Hartree)
        """
        vib_energy_D = (E - self.Morse_Dp.energy(0)) - (self.Morse_D.energy(vD) - self.Morse_D.energy(0))
        return self.hbarOmega(electronE) - (self.IP_D + vib_energy_D)

    def xs_bc(self, electronE:float, R:float, vD:int, E:float, FC_bc:float=None, norm:float=None) -> float:
        '''Cross section [a.u.] for one bound-continuum (bc) vibrational transition vi -> E.
        - E [Hartree] : energy of the dissociative Morse state
        - electronE [Hartree] : kinetic energy of the incoming electron
        - FC_bc [a.u.] : |<psi_E|psi_v>|^2
        '''
        if FC_bc is None:
            FC_bc = self.FC_bc(vD, E, norm=norm)
        electronE_f = self.electronE_f_bc(electronE, vD, E)
            
        if electronE_f <= 0:
            return 0
        else:
            hbarOmega = self.hbarOmega(electronE)
            PI_xs_A = self.PI_xs_A(hbarOmega)
            PI_xs_D = self.PI_xs_D_electronic(hbarOmega)
            xs = (
                self.prefactor
                * self.degeneracyFactor
                * PI_xs_A
                * PI_xs_D
                * FC_bc
                / (electronE * hbarOmega**2 * R**6)
            )
            return float(xs)
    
    # ====== OTHER ======
    
    def PR_xs_A(self, electronE):
        hbarOmega = self.hbarOmega(electronE)
        PI_xs = self.PI_xs_A(hbarOmega)
        return self.degeneracyFactor * hbarOmega**2 / (2*electronE*Constants.c**2) * PI_xs

    def plot_xs(self, ax, xs, label="ICEC", title='ICEC Cross section', **kwargs):
        """Plots the Cross section xs [Mb]"""
        ax.plot(self.energyGrid*Units.HARTREE2EV, xs, label=label, **kwargs)
        ax.set_xlabel(r'$E_\text{el}$ [eV]')
        ax.set_ylabel(r'$\sigma$ [Mb]')
        ax.set_yscale('log')
        ax.set_title(title)

    def plot_xs_R(self, ax, xs, **kwargs):
        """Plots the Cross section xs [Mb]"""
        ax.plot(self.rGrid, xs, **kwargs)
        ax.set_xlabel(r'$R$ [a.u.]')
        ax.set_ylabel(r'$\sigma$ [Mb]')
        ax.set_yscale('log')
        ax.set_title('ICEC cross section')

    def plot_PR_xs(self, ax, **kwargs):
        """Plots the Photorecombination Cross section [Mb]"""
        PR_xs = np.array([])
        for electronE in self.energyGrid:
            hbarOmega = electronE + self.IP_A
            PI_xs = self.PI_xs_A(hbarOmega)
            xs = self.degeneracyFactor * hbarOmega**2 / (2*electronE*Constants.c**2) * PI_xs
            PR_xs = np.append(PR_xs, [xs * Units.AU2MB])
        ax.plot(self.energyGrid*Units.HARTREE2EV, PR_xs, **kwargs)