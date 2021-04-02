# %%
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import root
from functools import partial
import numdifftools as nd
from bdahlen import makePlots, Potential
import warnings
warnings.filterwarnings("ignore")
# %%
if __name__ == '__main__':
    m = Potential(a = 100, nmax = 3)
    makePlots(m, [1, 2], 10)
# %%
    Vcut = m.cut(2, 0.)
    plt.plot(Vcut[0],Vcut[2])
# %%
