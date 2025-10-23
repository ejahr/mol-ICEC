import numpy as np
import copy
from .constants import Units, Constants

# ==========================================================
# ============= Asymptotic ICEC cross section ==============
# ==========================================================
class ICEC:
    """Electronic ICEC cross section.

    Uses atomic units (hbar=1, me=1, hartree energy=1).

    Args:
        degeneracyFactor(float): degeneracy of A- divided by degeneracy of A. g_{A^-} / g_A
        IP (float): Ionization potential (Hartree)
        PI_xs (float -> float): Fit for Photoionization cross section (a.u. -> a.u.)

    Attributes:
        degeneracyFactor(float): degeneracy of A- divided by degeneracy of A. g_{A^-} / g_A
        IP (float): Ionization potential (Hartree)
        PI_xs (float -> float): Fit for Photoionization cross section (a.u. -> a.u.)
        thresholdEnergy (float): Minimum kinetic energy of the incoming electron for ICEC to happen
        prefactor (float): collects terms that are neither energy nor R dependent 
        
    TODO take input in atomic units (IP, PI_xs)
    """  
    def __init__(self, degeneracyFactor:float, IP_A:float, IP_B:float, PI_xs_A, PI_xs_B) :
        self.degeneracyFactor = degeneracyFactor
        self.IP_A = IP_A * Units.EV2HARTREE
        self.IP_B = IP_B * Units.EV2HARTREE
        self.PI_xs_A = PI_xs_A
        self.PI_xs_B = PI_xs_B
        
        self.thresholdEnergy = max(0, self.IP_B - self.IP_A)
        self.prefactor = (3 * Constants.c**2) / (8 * np.pi)

    def make_energy_grid(self, minEnergy=None, maxEnergy=10, num:int=100, geometric=True): 
        """ Generates a suitable grid of incoming electron energies.
        - Energy (eV)
        - num (int): number of grid points
        TODO add function where you can define the energy grid directly
        """
        maxEnergy = maxEnergy * Units.EV2HARTREE
        if minEnergy is None:
            minEnergy = self.thresholdEnergy
        else:
            minEnergy = minEnergy * Units.EV2HARTREE        
        if geometric:
            self.energyGrid = np.geomspace(minEnergy, maxEnergy, num)
        else:
            self.energyGrid = np.linspace(minEnergy, maxEnergy, num)

    def make_R_grid(self, Rmin=2, Rmax=10, num:int=100): 
        """ Generates a grid of interatomic distances.
        - R (float): Interatomic distance (Bohr)
        - num (int): number of grid points
        """
        self.rGrid = np.linspace(Rmin, Rmax, num)
        
    def hbarOmega(self, electronE:float) -> float:
        return electronE + self.IP_A 
    
    def electronE_f(self, electronE:float) -> float:
        return self.hbarOmega(electronE) - self.IP_B 

    # ----- CROSS SECTION -----    
    def xs(self, electronE:float, R:float) -> float:
        """ Calculates cross section (a.u.) of ICEC for some kinetic energy and R.
        - electronE (float): kinetic energy of incoming electron (Hartree, a.u.)
        - R (float): interatomic distance (Bohr)
        """   
        if electronE < self.thresholdEnergy: 
            return 0
        else: 
            hbarOmega = self.hbarOmega(electronE)
            PI_xs_A = self.PI_xs_A(hbarOmega*Units.HARTREE2EV)*Units.MB2AU
            PI_xs_B = self.PI_xs_B(hbarOmega*Units.HARTREE2EV)*Units.MB2AU
            return self.prefactor * self.degeneracyFactor * PI_xs_A * PI_xs_B / (electronE * hbarOmega**2 * R**6)

    def xs_energy(self, R:float):
        """ Calculates cross section (Mb) of ICEC for given range of kinetic energies.
        - R (float): interatomic distance (Bohr)
        """        
        if not hasattr(self, 'energyGrid'):
            self.make_energy_grid()
        xs = np.array([
            self.xs(energy, R)
            for energy in self.energyGrid
        ]) 
        return xs * Units.AU2MB

    def xs_R(self, electronE:float):
        """ Calculates cross section (Mb) of ICEC for given range of interatomic distances R.
        - electronE (float): energy of incoming electron (Hartree) 
        """
        if not hasattr(self, 'rGrid'):
            self.make_R_grid()
        xs = np.array([
            self.xs(electronE, R)
            for R in self.rGrid
        ])
        return xs * Units.AU2MB

    def plot_xs(self, ax, xs, label="ICEC", **kwargs):
        """Plots the Cross section xs [Mb]"""
        ax.plot(self.energyGrid*Units.HARTREE2EV, xs, label=label, **kwargs)
        ax.set_xlabel(r'$E_\text{el}$ [eV]')
        ax.set_ylabel(r'$\sigma$ [Mb]')
        ax.set_yscale('log')
        ax.set_title('ICEC cross section')

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
            PI_xs = self.PI_xs_A(hbarOmega*Units.HARTREE2EV)*Units.MB2AU
            xs = self.degeneracyFactor * hbarOmega**2 / (2*electronE*Constants.c**2) * PI_xs
            PR_xs = np.append(PR_xs, [xs * Units.AU2MB])
        mask = PR_xs>0
        ax.plot(self.energyGrid[mask]*Units.HARTREE2EV, PR_xs[mask], **kwargs)
        
        
# ==========================================================
# ============= Asymptotic ICEC cross section ==============
# ==== for the Overlap (electron transfer) contribution ====
# ==========================================================
class OverlapICEC(ICEC):
    @classmethod
    def from_ICEC(cls, InstanceICEC: ICEC):
        """ Generates an instance of OverlapICEC from an instance of ICEC"""
        new_inst = copy.deepcopy(InstanceICEC) 
        new_inst.__class__ = cls
        return new_inst
    
    def define_overlap_parameters(self, a_A:float, a_B:float, C:float, d:float, lmax:int=10, gaussian_type:str='s'):
        """ Defines the overlap parameters, including fitting parameters
        - lmax: upper bound for sum over l -> set large enough for convergence
        """
        self.a_A = a_A
        self.a_B = a_B
        self.C = C
        self.d = d
        self.lmax = lmax
        self.gaussian_type = gaussian_type

    def Sab(self, R:float) -> float:
        """ Square of the overlap integral of two Gaussians
        """
        a_AB = self.a_A**2 + self.a_B**2
        if self.gaussian_type == 'pz':
            factor = 16 * self.a_A**3 * (self.a_B / a_AB)**5 * R**2
        elif self.gaussian_type == 's':
            factor = (2 * self.a_A * self.a_B / a_AB)**3
        else:
            print('Invalid gaussian type')
            return 0
        return factor* np.exp(-R**2/a_AB)

    def xs(self, electronE:float, R:float) -> float:
        """ Calculates cross section (a.u.) of the overlap contribution.
        - electronE (float): kinetic energy of incoming electron (Hartree)
        - R (float): internuclear distance (Bohr)
        """ 
        electronE_f = electronE + self.IP_A - self.IP_B
        if electronE_f <= 0 :
            return 0
        else: 
            # overlap of the continuum electrons
            C = self.C * np.exp(-abs(self.IP_A-self.IP_B)/self.d)
            sum_l = 0
            for l in range(0,self.lmax+1):  # noqa: E741
                K_av = electronE*(self.a_A+R)**2 + electronE_f*(self.a_B+R)**2
                J_l = np.exp(-l*(l+1)/K_av)
                sum_l += (2*l+1) * J_l
            sum_l *= C
            # cross section
            return 4*np.pi / electronE**(3/2) / np.sqrt(electronE_f) / R**2 * self.Sab(R) * sum_l