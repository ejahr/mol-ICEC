import numpy as np
import mpmath
from .constants import Constants, Units
from typing import Callable

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
        self.mu = mu  # in electron mass
        self.we = we # a.u.
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
        return self.De * (1 - np.exp(-self.alpha * (r - self.re))) ** 2 #- self.De

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
            float: Energy value of the v-th bound state (Hartree).
        """
        if self.wexe > 0:
            return self.we * (v + 0.5) - self.wexe * (v + 0.5) ** 2 
        return self.we * (v + 0.5) - (self.we * (v + 0.5)) ** 2 / (4 * self.De) #- self.De

    def intersection_V(self, E:float) -> float:
        arg = (-self.De + np.sqrt(self.De**2 + self.De * E)) / E
        return self.re + np.log(arg) / self.alpha

    def make_rgrid(self, resolution=1000, rmin=None, rmax=None):
        """Generates a grid of interatomic distances r (Bohr, a.u.)
        - resolution : number of grid points
        """
        if rmin is None:
            rmin = self.rmin
        if rmax is None:
            rmax = self.rmax
        self.r = np.linspace(rmin, rmax, resolution)
        return self.r

    def plot_V(self, ax, **kwargs):
        """ Plots the potential energy surface V(r)"""
        if not hasattr(self, "r"):
            self.make_rgrid()
        V = self.V(self.r)
        ax.set_xlabel(r"$R$ [a.u.]")
        ax.set_ylabel(r"$E$ [a.u.]")
        ax.plot(self.r, V, **kwargs)