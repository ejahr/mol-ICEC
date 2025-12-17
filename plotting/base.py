import matplotlib.pyplot as plt

DIR = '/home/elena/intraICEC/dimers/'

def set_rcParams():
    plt.rcdefaults()
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['axes.axisbelow'] = True