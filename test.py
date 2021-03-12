import numpy as np
import matplotlib.pyplot as plt

from cosmoTransitions import pathDeformation as pd


class Potential:
    def __init__(self, a = 50., E = 0., epsilon = .1, nmax = 5):
        self.params = a, E, epsilon, nmax

    def V(self, X):
        x,y = X[...,0], X[...,1]
        a, E, epsilon, nmax = self.params
        somatory = sum([(1 - E - n*epsilon)*np.exp(-a*np.power(x - n, 2)) for n in np.arange(nmax)])
        return 1/np.power(y, 2) - np.exp(-a*np.power(y - 1, 2))*somatory

    def dV(self, X):
        x,y = X[...,0], X[...,1]
        a, E, epsilon, nmax = self.params
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
        a, E, epsilon, nmax = self.params
        xmin, xmax, ymin, ymax = 0, nmax, 0.5, 1.5
        X = np.linspace(xmin, xmax,nx)[:,None] * np.ones((1,nx))
        Y = np.linspace(ymin, ymax,nx)[None,:] * np.ones((nx,1))
        XY = np.rollaxis(np.array([X,Y]), 0, 3)
        Z = self.V(XY)
        plt.contour(X,Y,Z, np.linspace(np.min(Z), np.max(Z)*.3, 200),
                    linewidths=0.5)


def makePlots():
    # Thin-walled instanton
    plt.figure()
    ax = plt.subplot(221)
    m = Potential()
    m.plotContour()
    # Y = pd.fullTunneling([[0.,1.],[2.,1.]], m.V, m.dV)
    # ax.plot(Y.Phi[:,0], Y.Phi[:,1], 'k', lw=1.5)
    # ax.set_xlabel(r"$\phi_x$")
    # ax.set_ylabel(r"$\phi_y$")
    # ax.set_title("Thin-walled")
    # ax = plt.subplot(223)
    # ax.plot(Y.profile1D.R, Y.profile1D.Phi, 'r')
    # ax.set_xlabel("$r$")
    # ax.set_ylabel(r"$|\phi(r) - \phi_{\rm absMin}|$")

    plt.show()

if __name__ == "__main__":
    makePlots()