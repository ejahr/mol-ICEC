import numpy as np
from .constants import Units, Constants

# ==========================================================
# ============= Asymptotic ICEC cross section ==============
# ==========================================================
class ICEC:
    '''Electronic ICEC cross section. See Jahr2026, https://doi.org/10.1063/5.0333097

    Uses atomic units (hbar=1, me=1, hartree energy=1).

    Attributes:
        degeneracyFactor (float): degeneracy of A- divided by degeneracy of A. g_{A^-} / g_A
        IP (float): Ionization potential (Hartree)
        PI_xs (float -> float): Fit for Photoionization cross section (a.u. -> a.u.)
        thresholdEnergy (float): Minimum kinetic energy of the incoming electron for ICEC to happen
        prefactor (float): collects terms that are neither energy nor R dependent 
        
    '''  
    def __init__(self, degeneracyFactor:float, IP_A:float, IP_D:float, PI_xs_A, PI_xs_D) :
        '''
        Args:
            degeneracyFactor (float): degeneracy of A- divided by degeneracy of A. g_{A^-} / g_A
            IP (float): Ionization potential (a.u., Hartree)
            PI_xs (float -> float): Fit for Photoionization cross section (a.u. -> a.u.)    
        '''
    
        self.degeneracyFactor = degeneracyFactor
        self.IP_A = IP_A
        self.IP_D = IP_D
        self.PI_xs_A = PI_xs_A
        self.PI_xs_D = PI_xs_D
        
        self.thresholdEnergy = max(0, self.IP_D - self.IP_A)
        self.prefactor = ( 3 * Constants.c**4 ) / ( 4 * np.pi )
        
    # ====== GRIDS ======

    def make_energy_grid(self, minEnergy=None, maxEnergy=10*Units.EV2HARTREE, num:int=100, geometric=True): 
        ''' Generates a suitable grid of incoming electron energies.
        - Energy (a.u./Hartree)
        - num (int): number of grid points
        TODO add function where you can define the energy grid directly
        '''
        if minEnergy is None:
            minEnergy = max(0.01*Units.EV2HARTREE, self.thresholdEnergy) 
        if geometric:
            self.energyGrid = np.geomspace(minEnergy, maxEnergy, num)
        else:
            self.energyGrid = np.linspace(minEnergy, maxEnergy, num)
        
    # ====== ENERGY RELATIONS ======
        
    def omega(self, electronE:float) -> float:
        '''omega = electronE + IP_A  [Hartree]
        Eq. (5) in Jahr2026, https://doi.org/10.1063/5.0333097
        '''
        omega = electronE + self.IP_A 
        if omega == 0:
            raise ZeroDivisionError('omega must not be zero')
        return omega
    
    def electronE_f(self, electronE:float) -> float:
        "electronE_f = omega - IP_D  [Hartree]"
        return self.omega(electronE) - self.IP_D 
    
    def PR_xs_A(self, electronE):
        "Eq. (39) in Jahr2026, https://doi.org/10.1063/5.0333097"
        if electronE == 0:
            raise ZeroDivisionError('electronE must not be zero')
        omega = self.omega(electronE)
        PI_xs = self.PI_xs_A(omega)
        return self.degeneracyFactor * omega**2 / ( 2 * electronE * Constants.c**2 ) * PI_xs

    # ====== CROSS SECTION ======  
       
    def xs(self, electronE:float, R:float) -> float:
        ''' Calculates cross section (a.u.) of ICEC for some kinetic energy and R.
        - electronE (float): kinetic energy of incoming electron (Hartree, a.u.)
        - R (float): interatomic distance (Bohr)
        Eq. (4) in Jahr2026, https://doi.org/10.1063/5.0333097
        '''   
        if electronE < self.thresholdEnergy: 
            return 0
        else: 
            omega = self.omega(electronE)
            PR_xs_A = self.PR_xs_A(electronE)
            PI_xs_D = self.PI_xs_D(omega)
            return self.prefactor * PR_xs_A * PI_xs_D / ( omega**4 * R**6 )

    def xs_energy(self, R:float):
        ''' Calculates cross section (Mb) of ICEC for given range of kinetic energies.
        - R (float): interatomic distance (Bohr)
        '''        
        if not hasattr(self, 'energyGrid'):
            self.make_energy_grid()
        xs = np.array([
            self.xs(energy, R)
            for energy in self.energyGrid
        ]) 
        return xs
    
    # ====== PLOTS ======
    
    def set_axes(ax):
        '''
        x label : epsilon \n
        y label : sigma \n
        enables log axis and grid
        '''
        ax.set_yscale('log')
        ax.set_xlabel(r'$\varepsilon$ [eV]')
        ax.set_ylabel(r'$\sigma$ [Mb]')
        ax.grid(True)

    def plot_xs(self, ax, R=None, xs=None, label=None, **kwargs):
        '''Plots the Cross section xs [Mb] against incoming electron energies [eV]'''
        if xs is None and R is not None:
            xs = self.xs_energy(R)
        xs *= Units.AU2MB
        energy = self.energyGrid * Units.HARTREE2EV
        if label is None:
            label = "icec"
            self.set_axes(ax)
        ax.plot(energy, xs, label=label, **kwargs)

    def plot_PR_xs(self, ax, **kwargs):
        '''Plots the Photorecombination Cross section [Mb]'''
        PR_xs = np.array([
            self.PR_xs_A(electronE) for electronE in self.energyGrid
        ])
        mask = PR_xs>0
        energy = self.energyGrid[mask]*Units.HARTREE2EV
        xs = PR_xs[mask]*Units.AU2MB
        ax.plot(energy, xs, **kwargs)