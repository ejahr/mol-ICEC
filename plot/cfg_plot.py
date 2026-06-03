import matplotlib.pyplot as plt

width, height = 6, 4

# spectrum
bar_width = 0.002 # LiH
bar_width = 0.02 # O2

# PES
height_ratios = [0.3, 0.7]      # LiH
height_ratios = [0.54, 0.46]      # O2
num_final_bound_states = 3
show_continuum_states = False
scale = 1./15
yshift = 7 # LiH
yshift = 0

def set_rcParams():
    plt.rcdefaults()
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['axes.axisbelow'] = True