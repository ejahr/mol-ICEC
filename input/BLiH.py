import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from input.fit import generate_linfit
from input.HLiH import LiH
from icec.constants import Units

DIR = '/home/elena/intraICEC/dimers/'

# =================== B+ ==========================
class B:
    deg_2P = 6
    deg_1S = 1
    deg_factor = 6
    IP = 8.298019 * Units.EV2HARTREE

    fname = DIR + 'data/B/B.txt'
    PI_xs = generate_linfit(fname)
    
    
# ===================== B+ = LiH =================
class Bp_LiH:
    input = [B.deg_factor, B.IP, LiH.IP, B.PI_xs, LiH.file_PI_xs_resolved]
    