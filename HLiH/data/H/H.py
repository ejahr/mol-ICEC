'''
Collects (additional) data for Hydrogen
- Ionization potentials
- mass
Not needed for the calculation of ICEC. All necessary parameters are defined in config.py
'''

from icec.constants import Units, Constants

class ReadOnly(type):
    def __setattr__(self, name, value):
        raise AttributeError("Constants are read-only")
    
class H(metaclass=ReadOnly):
    # H 2S_1/2
    deg_2S = 2
    deg_factor = deg_2S / 1

    # NIST
    IP = 13.598434599702 * Units.EV2HARTREE     # Rydberg R_H \approx IP
    m = 1.00782503223 * Constants.m_p
    
    r_vdw = 3.1647 # a.u.