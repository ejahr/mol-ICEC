import numpy as np
import scipy as sp
import mpmath
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from .constants import Units

class Morse:
    """Morse potential model for diatomic molecules.

    Uses atomic units (hbar=1, me=1, hartree energy=1).
    Adapted from https://scipython.com/blog/the-morse-oscillator and https://liu-group.github.io/Morse-potential

    Args:
        mu (float): Reduced mass (electron mass).
        we (float): Vibrational constant (Hartree).
        re (float): Equilibrium bond distance (Bohr).
        De (float): Dissociation energy (Hartree).
        wexe (float, optional): Defaults to 0.
        E0 (float, optional): Reference energy offset (Hartree). Defaults to 0.

    Attributes:
        mu (float): Reduced mass (electron mass).
        we (float): Vibrational constant (Hartree).
        re (float): Equilibrium bond distance (Bohr).
        De (float): Dissociation energy (Hartree).
        alpha (float): Morse alpha parameter.
        lam (float): Morse lambda parameter. sqrt(2 * mu * De) / alpha
        z0 (float): 2 * lam * exp(alpha * re)
        vmax (int): Maximum bound vibrational quantum number.
        rmin (float): Suggested lower radial bound (Bohr).
        rmax (float): Suggested upper radial bound (Bohr).
        
    TODO put Morse class in separate file
    """

    def __init__(self, mu:float, we:float, re:float, De:float, wexe:float=0, E0:float=0):
        self.mu = mu
        self.we = we
        self.re = re 
        self.De = De 
        self.wexe = wexe
        self.E0 = E0 
        
        if wexe > 0:
            De = self.we**2 / 4 / self.wexe
            print('De', self.De, De)
            self.De = self.we**2 / 4 / self.wexe

        self.alpha = self.we * np.sqrt(self.mu / 2 / self.De)
        self.lam = np.sqrt(2 * self.mu * self.De) / self.alpha
        self.z0 = 2 * self.lam * np.exp(self.alpha * self.re)
        self.vmax = int(self.lam - 0.5)

        self.rmin = self.re - np.log(2) / self.alpha
        f = 0.99
        self.rmax = self.re - np.log(1 - f) / self.alpha

    def V(self, r:float) -> float:
        """Morse potential V(r), with V(inf) = 0.
        
        Args:
            r (float): Interatomic distance (Bohr).

        Returns:
            float: Potential energy (Hartree).
        """
        return self.De * (1 - np.exp(-self.alpha * (r - self.re))) ** 2 - self.De

    def psi(self, v:int, r:float):
        """Return the v-th bound-state eigenfunction at distance r.

        Args:
            v (int): Vibrational quantum number (0 <= v <= vmax).
            r (float): Interatomic distance (Bohr).

        Returns:
            mpmath.mpf or complex: Wavefunction value at r.
        """
        z = self.z0 * mpmath.exp(-self.alpha * r)
        N = mpmath.sqrt(
            (2 * self.lam - 2 * v - 1)
            * mpmath.factorial(v)
            * self.alpha
            / mpmath.gamma(2 * self.lam - v)
        )
        return (
            N
            * z ** (self.lam - v - 0.5)
            * mpmath.exp(-z / 2)
            * mpmath.laguerre(v, 2 * self.lam - 2 * v - 1, z)
        )

    def energy(self, v:int) -> float:
        """Return the energy of the v-th bound state.

        Args:
            v (int): Vibrational quantum number (0 <= v <= vmax).

        Returns:
            float: Energy of the v-th bound state, E < 0 (Hartree).
        """
        if self.wexe > 0:
            return self.we * (v + 0.5) - self.wexe * (v + 0.5) ** 2 - self.De
        return self.we * (v + 0.5) - (self.we * (v + 0.5)) ** 2 / (4 * self.De) - self.De

    def intersection_V(self, E:float) -> float:
        arg = (-self.De + np.sqrt(self.De**2 + self.De * E)) / E
        return self.re + np.log(arg) / self.alpha
    
    def reflection_point_left(self, E:float) -> float:
        '''Returns r where E = V(r), r in (0, Req]
        - E : energy (Hartree, a.u.)
        '''
        E = mpmath.convert(E)
        arg = 1 + np.sqrt((E + self.De)/self.De)
        #if not isinstance(E, float):
        #    raise TypeError(f'E is of type {type(E)} but should be of type float.')
        if arg == mpmath.mpf('0'):
            raise ValueError(f'Invalid Value: arg={E} in log(arg)')
        return self.re - mpmath.log(arg) / self.alpha
    
    def reflection_point_right(self, E:float) -> float:
        '''Returns r where E = V(r), r in [Req, infty)
        - E : energy (Hartree, a.u.)
        '''
        if E >= 0:
            raise ValueError("Energy E must be negative.")
        arg = 1 - np.sqrt((E + self.De)/self.De)
        return self.re - mpmath.log(arg) / self.alpha

    def define_box(self, box_length:float = 10*Units.ANGSTROM2BOHR):
        '''Defines box length for discretizing the dissociative continuum
        - box_length : box length (bohr, a.u.)
        '''
        self.box_length = box_length

    def get_lower_bound(self, E:float, num:int=500) -> float:
        '''Lower bound for neglecting the diverging r->0 behaviour of the dissociative Morse states. 
        - E : energy (Hartree, a.u.)
        - num : number of sample points
        '''
        R = self.reflection_point_left(E)
        R_samples = np.linspace(R / 2, R, num=num)
        psi_samples = np.array(
            [  # psi_diss() does not work with np.array directly due to mpmath
                np.abs(self.psi_diss(E, r)) for r in R_samples
            ]
        )
        min_index = np.nanargmin(psi_samples)
        return R_samples[min_index]

    def estimate_oscillation(self, E:float, d:int=5) -> int:
        '''Estimate oscillation based on a particle in a box: E_n = n^2*pi^2/(2*m*L^2)
        - E : energy (Hartree, a.u.)
        - d : divide n by d to not have just one period per interval
        '''
        # 
        n = self.box_length * np.sqrt(2 * self.mu * E) / np.pi
        return round(n / d)

    def norm_diss(self, E:float, dps=15):
        '''Box normalization of the dissociative Morse states.
        - E : energy (Hartree, a.u.)
        - lower_bound : lower bound for the integration
        '''

        def integrand(r):
            return mpmath.conj(self.psi_diss(E, r)) * self.psi_diss(E, r)
        lower_bound = self.get_lower_bound(E)
        if dps==15 and hasattr(self, 'diss_energies'):
            if np.where(self.diss_energies==E)[0][0] == 0:
                dps = 50
                print('norm', dps)
        with mpmath.workdps(dps):
            norm = mpmath.quadsubdiv(integrand, [lower_bound, self.box_length], maxdegree=30)
        return 1 / mpmath.sqrt(mpmath.fabs(norm))

    def psi_diss(self, E:float, r:float):
        '''Dissociative (continuum) states of the Morse potential
        source: https://doi.org/10.1088/0953-4075/21/16/011
        mpmath.hyp1f1: https://mpmath.org/doc/current/functions/hypergeometric.html#hyp1f1
        - E : energy above dissociation limit (Hartree)
        - r : interatomic distance (Bohr)
        '''
        if E < 0:
            raise ValueError(f'E should be > 0. But E = {E}')
        k = mpmath.sqrt(2 * self.mu * E)
        epsilon = k / self.alpha
        s = self.lam - 0.5
        z = self.z0 * mpmath.exp(-self.alpha * r)  
        A = mpmath.gamma(-2j * epsilon) / mpmath.gamma(-s - 1j * epsilon)
        psi_in = (
            A
            * z ** (1j * epsilon)
            * mpmath.hyp1f1(-s + 1j * epsilon, 2j * epsilon + 1, z)
        )
        psi_out = (
            mpmath.conj(A)
            * z ** (-1j * epsilon)
            * mpmath.hyp1f1(-s - 1j * epsilon, -2j * epsilon + 1, z)
        )
        return mpmath.exp(-z / 2) * (psi_in + psi_out)
    
    def solve_root(self, max_energy, root_estimate, scale=1, dps=15):
        def psi_diss_L(E:float):
            if hasattr(E, "__len__"):
                E = E[0]
            if E > max_energy or E <= 0: # don't go looking beyond (0,max_energy]
                return 100
            else:
                return mpmath.fabs(self.psi_diss(E, self.box_length))
          
        def psi_float(E:float) -> float:
            return float(psi_diss_L(E))  

        rough_root = sp.optimize.fsolve(psi_float, root_estimate, xtol=1e-6)[0]
        with mpmath.workdps(dps):
            root = mpmath.findroot(psi_diss_L, rough_root, solver='newton', verify=False)
        return np.abs(root)
    
    def find_solutions_in_box(self, max_energy:float=1*Units.EV2HARTREE, num:int=500, file_path:str=None):
        """Finds allowed dissociative Morse states in a given box of self.box_length by solving psi(E,L) = 0 for E.
        """
        first_root = self.solve_root(max_energy, root_estimate=1e-10, scale=1e-3, dps=50)
        print("first root", first_root, mpmath.fabs(self.psi_diss(first_root, self.box_length)))
        root_estimates = np.geomspace(float(first_root), max_energy, num)
            
        t0 = time.perf_counter()
        roots = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.solve_root, max_energy, root) for root in root_estimates]
            for future in as_completed(futures):
                try:
                    roots.append(future.result())
                except Exception as e:
                    print("root solve failed", e)
        t1 = time.perf_counter()
        print("time for parallel root finding:", t1 - t0)
        roots = unique_mpf(np.array(roots), rtol=1e-8) # also sorts the array
        roots = np.array([
            E for E in roots if mpmath.fabs(self.psi_diss(E, self.box_length)) < mpmath.mpf('1e-8') #*self.norm_diss(E)
        ])
        
        if file_path is not None:
            header = "Energies [eV] of the continuum solutions to the Morse potential with psi(L)=0 where L = " + str(round(self.box_length*Units.BOHR2ANGSTROM)) + " Angstrom"
            np.savetxt(file_path, np.transpose(roots*Units.HARTREE2EV), fmt='%1.8e', header=header)
        self.diss_energies = roots
        return roots, root_estimates
    
    def get_density_of_states(self, energies=None):
        '''Density of states according to rho(E_i) = 2/(E_{i-2} - E_{i+1})
        - energies [Hartree] : energies of all possible dissociative states (in a box)
        '''
        if energies is None:
            if not hasattr(self, 'diss_energies'):
                self.find_solutions_in_box()
            energies = self.diss_energies
        density_of_states = np.zeros(len(energies))
        density_of_states[1:-1] = 2/(energies[2:]-energies[0:-2])
        density_of_states[0] = 1/(energies[1]-energies[0])
        density_of_states[-1] = 1/(energies[-1]-energies[-2])
        self.density_of_states = density_of_states
        return density_of_states
    
    def save_diss_states(self, file_path:str, max_energy:float=1*Units.EV2HARTREE, num:int=500):
        header = "Energies of the continuum solutions to the Morse potential with psi(L)=0 where L = " + str(round(self.box_length*Units.BOHR2ANGSTROM)) + " Angstrom\n"
        header += "E [eV] | E [a.u.] | norm [a.u.] | density of states [a.u.]"
        roots, root_estimates = self.find_solutions_in_box(max_energy, num)   
        
        t0 = time.perf_counter()
        with ProcessPoolExecutor() as executor:
            norms = list(executor.map(self.norm_diss, roots)) 
        t1 = time.perf_counter()
        print("time for parallel norm calculation:", t1 - t0)
        
        density_of_states = self.get_density_of_states() 
        data = np.vstack((roots*Units.HARTREE2EV, roots, norms, density_of_states))
        np.savetxt(file_path, np.transpose(data), fmt='%1.8e', header=header)
        return roots, root_estimates
        
    def load_diss_states(self, file_path:str):
        # TODO can I use dict for this?
        data = np.loadtxt(file_path, comments='#')
        self.diss_energies = data[:,1]
        self.diss_norms = data[:,2]
        self.density_of_states = data[:,3]

    def make_rgrid(self, num:int=1000, rmin:float=None, rmax:float=None):
        """Generates a grid of interatomic distances r (Bohr, a.u.)
        - resolution : number of grid points
        """
        if rmin is None:
            rmin = self.rmin
        if rmax is None:
            rmax = self.rmax
        self.r = np.linspace(rmin, rmax, num)
        return self.r

    def plot_V(self, ax, **kwargs):
        """ Plots the potential energy surface V(r)"""
        if not hasattr(self, "r"):
            self.make_rgrid()
        V = self.V(self.r)
        ax.set_xlabel(r"$R$ [a.u.]")
        ax.set_ylabel(r"$E$ [a.u.]")
        ax.plot(self.r, V, **kwargs)
        
        
def equal_mpf(a, b, rtol=1e-5, atol=0.0):
    if mpmath.fabs(a-b) <= atol + rtol * b:
        return True
    return False
        
def unique_mpf(arr, rtol=1e-5, atol=0.0):
    """ Returns sorted array of unique floats     
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    arr_sorted = np.unique(arr) # pre sort
    unique_list = [arr_sorted[0]]

    for x in arr_sorted[1:]:
        if not equal_mpf(x, unique_list[-1], rtol, atol):
            unique_list.append(x)

    return np.array(unique_list)