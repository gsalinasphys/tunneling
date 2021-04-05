import sys, os
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import root
from functools import partial
from cosmoTransitions import pathDeformation as pd

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

class Potential:
    def __init__(self, a = 50., E = 0., epsilon = .1, nmax = 5, whichmin = 2):
        self.params = a, E, epsilon, nmax, whichmin

    def V(self, X):
        x,y = X[...,0], X[...,1]
        a, E, epsilon, nmax = self.params[:-1]
        somatory = sum([(1 - E - n*epsilon)*np.exp(-a*np.power(x - n, 2)) for n in np.arange(nmax)])
        return 1/np.power(y, 2) - np.exp(-a*np.power(y - 1, 2))*somatory

    def dV(self, X):
        x,y = X[...,0], X[...,1]
        a, E, epsilon, nmax = self.params[:-1]
        somatory = sum([(1 - E - n*epsilon)*np.exp(-a*np.power(x - n, 2)) for n in np.arange(nmax)])
        dsomatory = sum([(1 - E - n*epsilon)*np.exp(-a*np.power(x - n, 2))*(-2*a*(x - n)) for n in np.arange(nmax)])
        dVdx = - np.exp(-a*np.power(y - 1, 2))*dsomatory
        dVdy = -2/np.power(y, 3) - np.exp(-a*np.power(y - 1, 2))*(-2*a*(y - 1))*somatory
        rval = np.empty_like(X)
        rval[...,0] = dVdx
        rval[...,1] = dVdy
        return rval

    def plotContour(self):
        nx = 100
        nmax = self.params[-2]
        xmin, xmax, ymin, ymax = 0, nmax, 0.5, 5
        X = np.linspace(xmin, xmax,nx)[:,None] * np.ones((1,nx))
        Y = np.linspace(ymin, ymax,nx)[None,:] * np.ones((nx,1))
        XY = np.rollaxis(np.array([X,Y]), 0, 3)
        Z = self.V(XY)
        plt.contourf(X,Y,Z, np.round(np.linspace(np.min(Z), np.max(Z), 200), 2), cmap='RdGy')
        plt.colorbar()

    def realmin_meta(self):
        whichmin = self.params[-1]
        return root(nd.Gradient(self.V), [whichmin, 1.]).x

    def realmin_zero(self):
        return root(nd.Gradient(self.V), [0., 1.]).x

    def cut(self, coeff_ang, xcut): #cut starts from 'whichmin', coeff_ang is between 0 and +infty
        real_min = self.realmin_meta()
        ycut = -coeff_ang*(xcut - real_min[0]*np.ones_like(xcut)) + real_min[1]
        return self.V(np.array([xcut, ycut]).T) 

def makePlot(m, factor_steps = 10):    # whichmins is a list of positive integers, i.e. [1, 2, 3, 4] or [2]
    # blockPrint()
    
    plt.figure()
    m.plotContour()

    true, meta = m.realmin_zero(), m.realmin_meta()

    iniradius = (meta[0] - true[0])/2
    inipath_x = np.linspace(true[0], meta[0])

    factor_range = np.linspace(0, 2, factor_steps, endpoint=True)
    Y, Yposs = None, None
    for factor in factor_range:
        inipath_y = np.sqrt(-np.power(inipath_x - iniradius - true[0],2) + np.power(iniradius,2))
        inipath_y = factor*inipath_y + true[1]
        inipath = np.array([inipath_x, inipath_y]).T
        try:
            Yposs = pd.fullTunneling(inipath, m.V, m.dV, maxiter=2)
        except Exception as e:
            enablePrint()
            print(e)
            blockPrint()
            pass

        if Yposs is not None and Y is not None:
            if Yposs.action < Y.action:
                Y = Yposs
        elif Yposs is not None and Y is None:
            Y = Yposs

    if Y is not None:
        plt.plot(Y.Phi[:,0], Y.Phi[:,1], 'k', lw=1.5)