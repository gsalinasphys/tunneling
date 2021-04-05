# %%
import numpy as np 
import matplotlib.pyplot as plt
import numdifftools as nd
import warnings
import time
from scipy.optimize import root
from scipy.signal import argrelextrema
from sympy import Symbol
from functools import partial
from bdahlen import makePlot, Potential
from cosmoTransitions.tunneling1D import SingleFieldInstanton
warnings.filterwarnings("ignore")
# %%
# x = Symbol('x')
# def my_deriv(f):
#     fsmp = f(x)
#     return lambdify(x, fsmp.diff(x), 'numpy')
# %%
if __name__ == '__main__':
    m = Potential(a = 50, nmax = 3, whichmin = 1)
    makePlot(m)
# %%
    Vcut = lambda x: m.cut(0.0, x)
    # Vcut = partial(m.cut, 0)
    xplot = np.linspace(-1,3,1000)
    Vplot = Vcut(xplot)
    plt.plot(xplot, Vplot)
# %%
    whichmin = m.params[-1]
    mins1d_index = argrelextrema(Vplot, np.less)
    zero1d, meta1d = xplot[mins1d_index[0][0]], xplot[mins1d_index[0][whichmin]]
    # s = SingleFieldInstanton(zero1d, meta1d, Vcut, nd.Derivative(Vcut))
    # s = SingleFieldInstanton(zero1d, meta1d, Vcut, my_deriv(Vcut))
    s = SingleFieldInstanton(zero1d, meta1d, Vcut)
# %%
    start = time.process_time()
    pfl = s.findProfile()
    s.findAction(pfl)
    print((time.process_time() - start)/60.)
# %%
