import numpy as np
import mpmath
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
            return self.we * (v + 0.5) - self.wexe * (v + 0.5) ** 2 
        return self.we * (v + 0.5) - (self.we * (v + 0.5)) ** 2 / (4 * self.De) - self.De

    def intersection_V(self, E:float) -> float:
        arg = (-self.De + np.sqrt(self.De**2 + self.De * E)) / E
        return self.re + np.log(arg) / self.alpha
    
    def reflection_point_left(self, E:float) -> float:
        '''Returns r where E = V(r), r in (0, Req]
        - E : energy (Hartree, a.u.)
        '''
        arg = 1 + np.sqrt((E + self.De)/self.De)
        return self.re - np.log(arg) / self.alpha
    
    def reflection_point_right(self, E:float) -> float:
        '''Returns r where E = V(r), r in [Req, infty)
        - E : energy (Hartree, a.u.)
        '''
        if E >= 0:
            raise ValueError("Energy E must be negative.")
        arg = 1 - np.sqrt((E + self.De)/self.De)
        return self.re - np.log(arg) / self.alpha

    def define_box(self, box_length:float = 10*Units.ANGSTROM2BOHR):
        '''Defines box length for discretizing the dissociative continuum
        - box_length : box length (bohr, a.u.)
        '''
        self.box_length = box_length

    def get_lower_bound(self, E:float, num:int=200) -> float:
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

    def norm_diss(self, E:float, lower_bound:float=None) -> float:
        '''Box normalization of the dissociative Morse states.
        The integration is separated into intervals as these states can be highly-oscillating.
        - E : energy (Hartree, a.u.)
        - lower_bound : lower bound for the integration
        '''
        def integrand(r):
            return mpmath.conj(self.psi_diss(E, r)) * self.psi_diss(E, r)
        if lower_bound is None:
            lower_bound = self.get_lower_bound(E)
        r_reflection = self.reflection_point_left(E)
        num_intervals = self.estimate_oscillation(E)
        if num_intervals < 10:
            norm = mpmath.quadsubdiv(integrand, [lower_bound, r_reflection, self.rmax, self.box_length], maxdegree=10)
        else:
            norm = mpmath.quadsubdiv(integrand, [lower_bound, r_reflection], maxdegree=10)
            intervals_mid = np.linspace(r_reflection, self.rmax, num_intervals + 1)
            norm += mpmath.quadsubdiv(integrand, intervals_mid, maxdegree=10)
            intervals_high = np.linspace(self.rmax, self.box_length, num_intervals + 1)
            norm += mpmath.quadsubdiv(integrand, intervals_high, maxdegree=10)
        return 1 / mpmath.sqrt(norm)

    def psi_diss(self, E:float, r:float):
        '''Dissociative (continuum) states of the Morse potential
        source: https://doi.org/10.1088/0953-4075/21/16/011
        mpmath.hyp1f1: https://mpmath.org/doc/current/functions/hypergeometric.html#hyp1f1
        - E : energy above dissociation limit (Hartree)
        - r : interatomic distance (Bohr)
        '''
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