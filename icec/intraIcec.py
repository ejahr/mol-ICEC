import numpy as np
import scipy as sp
import mpmath
import time
import copy
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from typing import Callable
from .constants import Constants, Units
from .morse import Morse
from .icec import ICEC

class IntraICEC(ICEC):
    '''ICEC cross section with diatomic molecules. See Jahr2026, https://doi.org/10.1063/5.0333097

    Uses atomic units (hbar=1, me=1, hartree energy=1).
    
    Attributes:
        degeneracyFactor(float) : degeneracy of A- divided by degeneracy of A. g_{A^-} / g_A
        IP (float): Ionization potential (Hartree)
        PI_xs_A (float -> float): Fit for Photoionization cross section of A (a.u. -> a.u.)
        file_PI_xs_D (str) : file name of the photionization cross section of D
        prefactor (float): collects terms that are neither energy nor R dependent 
    '''  
    
    def __init__(self, degeneracyFactor: float, IP_A: float, IP_D: float, PI_xs_A: Callable, file_PI_xs_D: str) :
        '''
        Args:
            degeneracyFactor(float) : degeneracy of A- divided by degeneracy of A. g_{A^-} / g_A
            IP (float): Ionization potential (Hartree)
            PI_xs_A (float -> float): Fit for Photoionization cross section of A (a.u. -> a.u.)
            file_PI_xs_D (str) : file name of the photionization cross section of D
        '''
        
        self.degeneracyFactor = degeneracyFactor # g_A / g_A+
        self.IP_A = IP_A 
        self.IP_D = IP_D # assumption: adiabatic ionization energy
        self.PI_xs_A = PI_xs_A
        self.file_PI_xs_D = file_PI_xs_D
        self.prefactor = 3 * Constants.c**4 / ( 4 * np.pi )
        self.make_energy_grid()
       
    def define_Morse_D(self, mu:float, we:float, re:float, De:float, wexe:float=0):
        '''Morse potential for the PES of D.
        - mu: reduced mass (proton mass)
        - we: Morse parameter (a.u.)
        - re: Equilibrium bond distance (a.u.)
        - De: Dissociation energy (a.u.)
        - wexe : optional (a.u.)
        '''
        self.Morse_D:Morse = Morse(mu, we, re, De, wexe)

    def define_Morse_Dp(self, mu:float, we:float, re:float, De:float, wexe:float=0, box_length:float=10*Units.ANGSTROM2BOHR):
        '''Morse potential for the PES of D+.
        - mu: reduced mass (proton mass)
        - we: Morse parameter (a.u.)
        - re: Equilibrium bond distance (a.u.)
        - De: Dissociation energy (a.u.)
        - wexe : optional (a.u.)
        - box_length: size of the box for discretizing dissociative states
        '''
        self.Morse_Dp:Morse = Morse(mu, we, re, De, wexe)
        self.Morse_Dp.define_box(box_length)
        
    # ====== GRIDS ======

    def make_energy_grid(self, minEnergy=0.01*Units.EV2HARTREE, maxEnergy=10*Units.EV2HARTREE, num=100, geometric=True): 
        ''' Make a suitable grid of incoming electron energies.
        - minEnergy : minimum energy of the grid (a.u.)
        - maxEnergy : maximum energy of the grid (a.u.)
        - num : number of grid points
        '''
        if geometric:
            self.energyGrid = np.geomspace(minEnergy, maxEnergy, num)
        else:
            self.energyGrid = np.linspace(minEnergy, maxEnergy, num)
        
    # ====== PHOTOIONIZATION CROSS SECTION ======
        
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
    def PI_xs_D_resolved(self, vD:int, vDp:int, omega:float):
        filename = self.file_PI_xs_D + f"{vD}_{vDp}.txt"
        data = np.loadtxt(filename)
        energies, xs = data[:, 0]*Units.EV2HARTREE, data[:, 1]*Units.MB2AU
        if omega < energies[0]:
            return 0
        elif omega > energies[-1]:
            return np.nan
        else:
            interp_func = sp.interpolate.interp1d(energies, xs, kind='linear')
            return interp_func(omega)

    def PI_xs_D_FC(self, vD:int, vDp:int, omega:float) -> float:
        "Eq. (32) in Jahr2026, https://doi.org/10.1063/5.0333097"
        PI_xs_electronic = self.PI_xs_D_electronic(omega)
        if np.isnan(PI_xs_electronic):
            return np.nan
        else:
            return PI_xs_electronic * self.FC_bb_D(vD, vDp)
    
    def PI_xs_D_electronic(self, omega:float) -> float:
        data = np.loadtxt(self.file_PI_xs_D)
        energies, xs = data[:, 0]*Units.EV2HARTREE, data[:, 1]*Units.MB2AU
        if omega < energies[0]:
            return 0
        elif omega > energies[-1]:
            return np.nan
        else:
            interp_func = sp.interpolate.interp1d(energies, xs, kind='linear')
            return interp_func(omega) 
        
    # ====== FRANCK-CONDON FACTORS ======
    
    def FC_bb_D(self, vD:int, vDp:int) -> float:
        '''Returns the Franck-Condon factor <psi_vi|psi_vf> corresponding to the photoionization.
        - vD (int): vibrational quantum number of D.
        - vDp (int): vibrational quantum number of D+.
        '''
        if not hasattr(self, 'FC_factor_saved'):
            # vmax+1 results out of bound... temp fix: +10 
            self.FC_factor_saved = np.zeros((self.Morse_D.vmax+10, self.Morse_Dp.vmax+10))
        if self.FC_factor_saved[vD][vDp] > 0:
            return self.FC_factor_saved[vD][vDp]
        else:
            # TODO separate to integration function
            def integrand(r):
                return mpmath.conj(self.Morse_D.psi(vD, r)) * self.Morse_Dp.psi(vDp, r)
            r_left = min(self.Morse_D.reflection_point_left(self.Morse_D.energy(vD)), 
                         self.Morse_Dp.reflection_point_left(self.Morse_Dp.energy(vDp)))
            r_right = max(self.Morse_D.rmax, self.Morse_Dp.rmax)
            intervals = [0, r_left, r_right, 10*Units.ANGSTROM2BOHR]
            result = mpmath.quadsubdiv(integrand, intervals, maxdegree=10)
            self.FC_factor_saved[vD][vDp] = np.abs(result)**2
            return np.abs(result)**2
        
    def integrate_r(self, integrand, E):
        '''Integration over r
        - integrand : function to be integrated
        - E : energy of the dissociative Morse state [Hartree]
        '''
        lower_bound = self.Morse_Dp.get_lower_bound(E)
        upper_bound = self.Morse_Dp.box_length
        result = mpmath.quadsubdiv(integrand, [lower_bound, upper_bound], maxdegree=30)
        return result
 
    def FC_bc_D(self, vD:int, E:float, norm:float=None, dps=15):
        '''Franck-Condon (FC) factor for a bound to continuum (bc) transition |<psi_E|psi_v>|^2
        - vD: vibrational quantum number of D
        - E: energy of the dissociative of D+, i.e. energy above dissociation limit (Hartree)
        - norm: normalization constant for the vibrational continuum state
        '''
        if norm is None:
            norm = self.Morse_Dp.get_norm_diss(E)
        def integrand(r):
            return mpmath.conj(self.Morse_Dp.psi_diss(E, r)) * self.Morse_D.psi(vD, r)
        if dps==15 and hasattr(self.Morse_Dp, 'diss_energies'):
            if np.where(self.Morse_Dp.diss_energies==E)[0][0] == 0:
                dps = 50
        with mpmath.workdps(dps):
            result = self.integrate_r(integrand, E)    
        FC_factor = (mpmath.fabs(norm * result)) ** 2
        return FC_factor
        
    # ====== ENERGY RELATIONS ======
    
    def input_vib_spacing_D(self, vib_spacing_D, vib_spacing_Dp):
        self.vib_diff_to_v0_D = np.cumsum(vib_spacing_D)
        self.vib_diff_to_v0_Dp = np.cumsum(vib_spacing_Dp)
    
    def convert_minima_to_vertical_IP(self, IP_minima):
        IP_vertical = IP_minima + (self.Morse_Dp.V(self.Morse_D.re) + self.Morse_Dp.De)
        return IP_vertical
    
    def change_minima_to_adiabatic_IP(self):
        IP_adiabatic = self.IP_D - (self.Morse_D.energy(0) + self.Morse_D.De) + (self.Morse_Dp.energy(0) + self.Morse_Dp.De)
        self.IP_D = IP_adiabatic
    
    def electronE_f(self, electronE:float, vD:int, vDp:int) -> float:
        # Eq. (18) in Jahr2026, https://doi.org/10.1063/5.0333097
        if vDp is None:
            vib_energy_D = 0
        elif hasattr(self, 'vib_diff_to_v0_D'):
            vib_energy_D = self.vib_diff_to_v0_Dp[vDp] - self.vib_diff_to_v0_D[vD]
        else:
            vib_energy_D = (self.Morse_Dp.energy(vDp) - self.Morse_Dp.energy(0)) - (self.Morse_D.energy(vD) - self.Morse_D.energy(0))
        return self.omega(electronE) - (self.IP_D + vib_energy_D)
    
    def electronE_f_bc(self, electronE:float, vD:int, E:float) -> float:
        '''Kinetic energy of the outgoing electron
        - electronE: kinetic energy of the incoming electron (Hartree)
        - vD: vibrational quantum number of D
        - E: energy of the dissociative of D+, i.e. energy above dissociation limit (Hartree)
        Eq. (18) in Jahr2026, https://doi.org/10.1063/5.0333097
        '''
        vib_energy_D = (E - self.Morse_Dp.energy(0)) - (self.Morse_D.energy(vD) - self.Morse_D.energy(0))
        return self.omega(electronE) - (self.IP_D + vib_energy_D)

    # ====== CROSS SECTION ======
       
    def xs(self, electronE:float, R:float, vD:int=0, vDp:int=0) -> float:
        ''' ICEC cross section (a.u.) for given kinetic energy and R.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - R : distance between center of masses of A and D (Bohr, a.u.)
        - vD -> vDp
        Eq. (19) in Jahr2026, https://doi.org/10.1063/5.0333097
        '''   
        if self.electronE_f(electronE, vD, vDp) <= 0: 
            return 0
        else: 
            omega = self.omega(electronE)
            PR_xs_A = self.PR_xs_A(electronE)
            PI_xs_D = self.PI_xs_D(vD, vDp, omega)
            return self.prefactor * PR_xs_A * PI_xs_D / ( omega**4 * R**6 )

    def xs_vD_vDp(self, R:float, vD:int=0, vDp:int=0):
        ''' ICEC cross section (a.u.) for given range of kinetic energies.
        - R : distance between center of masses of A and D (Bohr, a.u.)
        - vD -> vDp
        '''        
        xs = np.array([
            self.xs(energy, R, vD, vDp)
            for energy in self.energyGrid
        ]) 
        return xs
    
    def xs_vD(self, R:float, vD:int, vDp_max:int=None):
        ''' ICEC Cross section [a.u.] for vi -> bound states over range of electron energies.
        - R : distance between center of masses of A and D (Bohr, a.u.)
        - vD -> bound states
        Eq. (21) first line in Jahr2026, https://doi.org/10.1063/5.0333097
        '''
        t0 = time.perf_counter()
        if vDp_max is None:
            vDp_max = self.Morse_Dp.vmax
        # Element-wise summation sum(list_of_arrays)
        xs = sum(
            self.xs_vD_vDp(R, vD, vDp) 
            for vDp in range(vDp_max + 1)
        )
        t1 = time.perf_counter()
        print(f'b-b xs vD={vD} : {round(t1-t0,2)} s')
        return xs
    
    def xs_boltzmann(self, R:float, t:float, vD_max:int, vDp_max:int=None):
        # add De to energy(vi) to get positive values, increasing numerical stability
        # Eq. (28) in Jahr2026, https://doi.org/10.1063/5.0333097
        if vDp_max is None:
            vDp_max = self.Morse_Dp.vmax
        norm = sum(
            np.exp(-(self.Morse_D.energy(vD)+self.Morse_D.De)/Constants.KB/t) 
            for vD in range(vD_max+1)
        )
        avg = sum(
            np.exp(-(self.Morse_D.energy(vD)+self.Morse_D.De)/Constants.KB/t) * self.xs_vD(R, vD, vDp_max)
            for vD in range(vD_max+1)
        )
        return avg/norm
    
    def spectrum(self, electronE:float, R:float, vD:int=0, vDp_max:int=None):
        ''' Cross sections [Mb] for vi -> bound states given some electron energy.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - R : distance between center of masses of A and D (Bohr, a.u.)
        - vD -> bound states
        Returns
        - vD, vDp, electronE_f [eV], xs [Mb]
        '''
        t0 = time.perf_counter()
        if vDp_max is None:
            vDp_max = self.Morse_Dp.vmax
        spectrum = []
        for vDp in range(vDp_max+1):
            electronE_f = self.electronE_f(electronE, vD, vDp)
            if electronE_f >= 0:
                xs = self.xs(electronE, R, vD, vDp)
                spectrum.append(
                    [vD, vDp, electronE_f * Units.HARTREE2EV, xs * Units.AU2MB]
                )
        t1 = time.perf_counter()
        print(f'b-b spectrum vD={vD} : {round(t1-t0,2)} s')
        return np.array(spectrum)
    
    # ====== DISSOCIATION OF D only FC model ======

    def xs_bc(self, electronE:float, R:float, vD:int, E:float, FC_bc_D:float=None, norm:float=None) -> float:
        '''Cross section [a.u.] for one bound-continuum (bc) vibrational transition vi -> E.
        - electronE [Hartree] : kinetic energy of the incoming electron
        - vD: vibrational quantum number of D
        - E: energy of the dissociative of D+, i.e. energy above dissociation limit (Hartree)
        - FC_bc [a.u.] : |<psi_E|psi_v>|^2
        
        Eq. (33) in Jahr2026, https://doi.org/10.1063/5.0333097
        '''
        if self.electronE_f_bc(electronE, vD, E) <= 0:
            return 0
        else:
            omega = self.omega(electronE)
            PR_xs_A = self.PR_xs_A(electronE)
            PI_xs_D = self.PI_xs_D_electronic(omega)
            if np.isnan(PI_xs_D):
                return np.nan
            if FC_bc_D is None:
                FC_bc_D = self.FC_bc_D(vD, E, norm=norm)
            return self.prefactor * PR_xs_A * PI_xs_D * FC_bc_D / ( omega**4 * R**6 )

    def xs_vD_E(self, R:float, vD:int, E:float):
        '''Cross section for vD -> E over range of electron energies.
        - R : distance between center of masses of A and D (Bohr, a.u.)
        - vD: vibrational quantum number of the initial state
        - E [Hartree] : energy of the dissociative Morse state
        '''
        FC_bc = self.FC_bc_D(vD, E)
        xs_array = np.array(
            [self.xs_bc(electronE, R, vD, E, FC_bc) for electronE in self.energyGrid]
        )
        return xs_array

    def xs_vD_continuum(self, R:float, vD:int, diss_energies=None, max_dissE=None):
        '''Cross section for vi -> continuum over range of electron energies.
        - R : distance between center of masses of A and D (Bohr, a.u.)
        - vD : vibrational quantum number of the initial state
        - diss_energies [Hartree] : energies of dissociative states (in a box)
        - max_dissE : maximum energy up to which the dissociative states are included
        
        Eq. (34) in Jahr2026, https://doi.org/10.1063/5.0333097
        '''   
        #if diss_energies is None:
        #    if not hasattr(self.Morse_Dp, 'diss_energies'):
        #        self.Morse_Dp.find_solutions_in_box()
        #    diss_energies = self.Morse_Dp.diss_energies
        if max_dissE is not None:
            mask = self.Morse_Dp.diss_energies <= max_dissE
            diss_energies = self.Morse_Dp.diss_energies[mask]
        t0 = time.perf_counter()
        with ProcessPoolExecutor() as executor:
            result = list(
                executor.map(
                    self.xs_vD_E, 
                    repeat(R), repeat(vD), diss_energies
                )
            )
        t1 = time.perf_counter()
        print(f'b-c xs vD={vD} : {round(t1-t0,2)} s')
        return sum(list(result))
    
    def function_for_spectrum(self, electronE, R, vD, E, density_of_states_at_E):
        electronE_f = self.electronE_f_bc(electronE, vD, E)
        if electronE_f <= 0:
            return electronE_f, 0, E
        else:
            # transform to energy normalization by multiplying with the density of states at E
            xs = self.xs_bc(electronE, R, vD, E) * Units.AU2MB * density_of_states_at_E / Units.HARTREE2EV
            return vD, E*Units.HARTREE2EV, electronE_f*Units.HARTREE2EV, xs

    def spectrum_bc(self, electronE, R, vD, diss_energies=None):
        '''Cross sections for vi -> continuum given a single electron energy.
        - electronE [Hartree] : kinetic energy of incoming electron 
        - R : distance between center of masses of A and D (Bohr, a.u.)
        - diss_energies [Hartree] : energies of dissociative states (in a box)
        Returns
        - vD, E [eV], electronE_f [eV], xs [Mb]
        '''
        if diss_energies is None:
            if not hasattr(self.Morse_Dp, 'diss_energies'):
                self.Morse_Dp.find_solutions_in_box()
            diss_energies = self.Morse_Dp.diss_energies
        if not hasattr(self.Morse_Dp, 'density_of_states'):
            self.Morse_Dp.get_DoS(diss_energies)
        t0 = time.perf_counter()
        with ProcessPoolExecutor() as executor:
            result = list(
                executor.map(
                    self.function_for_spectrum, 
                    repeat(electronE), repeat(R), repeat(vD), diss_energies, self.Morse_Dp.DoS
                )
            )
        t1 = time.perf_counter()
        print(f'b-c spectrum vD={vD} : {round(t1-t0,2)} s')
        return np.array(result)
        
        
class RydbergIntraICEC(IntraICEC):
    '''ICEC cross section for atom-molecule systems.

    Uses atomic units (hbar=1, me=1, hartree energy=1).

    Attributes:
        IP (float): Ionization potential (Hartree)
        PI_xs_A (float -> float): Fit for Photoionization cross section of A (a.u. -> a.u.)
        file_PI_xs_D (str) : file name of the photoionization cross section of D
        prefactor (float): collects terms that are neither energy nor R dependent 
    '''  
      
    @classmethod
    def from_IntraICEC(cls, InstanceICEC: IntraICEC, n:int):
        '''generate an instance of RydbergIntraICEC from an instance of IntraICEC 
        
        Args:
            IP (float): Ionization potential (Hartree)
            n (int) : principal number of the Rydberg state into which the electron is captured 
            file_PI_xs_D (str) : file name of the photoionization cross section of D
        '''
        
        new_inst = copy.deepcopy(InstanceICEC) 
        new_inst.__class__ = cls
        new_inst.n = n
        return new_inst
    
    def __init__(self, IP_A: float, IP_D: float, n:int, file_PI_xs_D: str) :
        self.IP_A = IP_A # assumption: IP_A = R_A Rydberg constant
        self.IP_D = IP_D # assumption: adiabatic ionization energy
        self.file_PI_xs_D = file_PI_xs_D
        self.n = n
        self.prefactor = 3 * Constants.c**4 / ( 4 * np.pi )
    
    def omega(self, electronE:float) -> float:
        return electronE + self.IP_A_n()
    
    def IP_A_n(self) -> float:
        ''' Rydberg formula E_n = E_1 / n^2'''
        return self.IP_A / self.n**2
    
    def PR_xs_A(self, electronE:float) -> float:
        ''' Quasiclassical photorecombination cross section for proton-like ions (Gokhberg 2010) 
        
        Eq. (40) in Jahr2026, https://doi.org/10.1063/5.0333097
        '''
        xs = 1.96 * np.pi**2 / Constants.c**3 \
            * self.IP_A**2 / ( electronE * ( electronE + self.IP_A_n() ) ) \
            * 1 / self.n**3
        return xs