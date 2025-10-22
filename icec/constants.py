import numpy as np

class ReadOnly(type):
    def __setattr__(self, name, value):
        raise AttributeError("Constants are read-only")

class Units(metaclass=ReadOnly):
    # energy
    EV2HARTREE = 3.67493e-2
    HARTREE2EV = 1/EV2HARTREE
    WAVENUMBER2HARTREE = 4.55633e-6 # cm^-1
    J2HARTREE = 2.2937104486906e17
    RYD2EV = 13.605693122990
    RYD2HARTREE = RYD2EV * EV2HARTREE
    
    # length
    BOHR2M = 5.29177210544e-11
    M2BOHR = 18897259885.789
    ANGSTROM2M = 1e-10
    ANGSTROM2BOHR = ANGSTROM2M * M2BOHR
    BOHR2ANGSTROM = 0.529177249
    PM2BOHR = 1e-12 * M2BOHR

    # cross section
    MB2M2 = 1e-22
    MB2AU = MB2M2 * M2BOHR**2
    AU2MB = BOHR2M**2 / MB2M2
    
    def angstrom2hartree(self, wavelength):
        # E = 2 * pi * hbar * c / lambda
        wavelength *= self.ANGSTROM2BOHR
        energy = 2 * np.pi * Constants.c / wavelength
        return energy

# physical constants in a.u.
class Constants(metaclass=ReadOnly):
    c = 137
    KB = 1.380649e-23 * Units.J2HARTREE # Hartree/K
    m_p = 1836.152673426 # electron mass
    
class Config(metaclass=ReadOnly):
    DEFAULT_BOX_LENGTH = 10*Units.ANGSTROM2BOHR 