# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:45:18 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Classes for graphical representation of MC simulation (Tissues geometry, photon paths...)
and post-processing data visualization
"""
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import MClayers
from MCPhaseFun import HenyeyGreenstein as HG  # debug

def find_closest(x,y,X,Y):
    """Return indices of closest value on 2D coordinates grid"""
    #TODO: this finds the immediate lower value. Compare to searchSorted +1 to find closest
    xi = np.searchsorted(X, x)
    if xi == len(X):
        xi -=1
    yi = np.searchsorted(Y, y)
    if yi == len(Y):
        yi -=1
    return xi,yi


class Geometries:
    """Class to visually show tissues and photons. Note: the view is in the xz plane"""
    def __init__(self, tissues=None):
        """
        Parameters
        ----------
        tissues : LIST of LAYER OBJECTS
            An ordered list containing all the layers in the geometry.

        Returns
        -------
        None.
        """
        self.tissues = tissues
        
    def loadGeometry(self, tissues):
        """
        Update the geometry
        
        Parameters
        ----------
        tissues : LIST of LAYER OBJECTS
            An ordered list containing all the layers in the geometry.

        Returns
        -------
        None.
        """
        self.tissues = tissues
        
    def showGeometry(self, xlim=[-10,10], zlim=[-10,10]):
        """
        Plot the geometry contained in self.tissues

        Returns
        -------
        ax : MATPLOTLIB AXIS OBJECT
        """
        plt.figure(num=1, figsize=(10,8))
        ax = plt.gca()
        for tissue in self.tissues[::-1]:  # draw tissues from the lowest
            if type(tissue) is MClayers.Infinite:  # outside tissue
                ax.set_facecolor(tissue.color)
            elif type(tissue) is MClayers.Slab:  # slab
                ax.axhspan(ymin=tissue.top, ymax=tissue.top+tissue.thickness, color=tissue.color)
        # draw light source
        ax.plot([0,0], [0, zlim[0]], linewidth=2, color='yellow', linestyle='-')
        
        # plt.axis('off')
        plt.xlim(xlim)
        plt.xlabel('mm')
        plt.ylim(zlim)
        plt.ylabel('mm')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
        return ax  # useful for other plots
    
    def showPaths(self, ax, photons, N=100, **kwargs):
        """
        Plots the geometry and shows photon paths

        Parameters
        ----------
        ax : MATPLOTLIB AXIS OBJECT     
        photons : LIST of simulated PHOTON OBJECTS
            Results of the simulation, with scattering paths
        N : INT
            number of photon paths to draw(first N)
        
        Returns
        -------
        None.
        """
        linewidth = kwargs.get('linewidth', 1)
        for photon in photons[:N]:
            photonPath = np.array([[x[0] for x in photon.path], [x[2] for x in photon.path]])
            ax.plot(photonPath[0], photonPath[1], linewidth=linewidth, linestyle='-', color='yellow', alpha=0.3)
    
    def animatePath(self, ax, photons, N=100, M=20, **kwargs):
        """
        Plot the photon paths and highlignt the first M

        Parameters
        ----------
        ax : MATPLOTLIB AXIS OBJECT
        photons : LIST of simulated PHOTON OBJECTS
            Results of the simulation, with scattering paths.
        N : INT, optional
            number of photon paths to draw(first N). The default is 100
        M : INT, optional
            The nuber of photons to highlight. The default is 20.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
        linewidth = kwargs.get('linewidth', 1)
        # First plot path
        for photon in photons[:N]:
            photonPath = np.array([[x[0] for x in photon.path], [x[2] for x in photon.path]])
            ax.plot(photonPath[0], photonPath[1], linewidth=linewidth, linestyle='-', color='yellow', alpha=0.3)
        # Now do the animation       
        photonPath = np.array([[x[0] for x in photons[0].path], [x[2] for x in photons[0].path]])
        current = ax.plot(photonPath[0], photonPath[1], linewidth=2,
                          linestyle='-',color='red')  # first one
        for _i, photon in enumerate(photons[1:M]):
            time.sleep(0.5)
            photonPath = np.array([[x[0] for x in photon.path], [x[2] for x in photon.path]])
            current[0].set_xdata(photonPath[0])
            current[0].set_ydata(photonPath[1])
            plt.draw()
            plt.pause(0.001)
    
    def showAbsorbed(self, absorbed, xlim=[-10,10], zlim=[-10,10], res=0.1):
        """
        Show colormap with photon absorption

        Parameters
        ----------
        absorbed : LIST of NUMPY ARRAYS
            List of coordinates/photon weigths -> (x,y,z,w)

        Returns
        -------
        None.
        """
        # Create meshgrid to interpolate. Default resolution is 0.1mm
        x = np.linspace(xlim[0], xlim[1], num=int((xlim[1]-xlim[0])/res) + 1, dtype=float)
        z = np.linspace(zlim[0], zlim[1], num=int((zlim[1]-zlim[0])/res) + 1, dtype=float)
        weights = np.ones((len(z), len(x)), dtype=float)*1e-16  # weights grid. Slightly larger than 0 for logNorm
        for point in absorbed:
            # find the closest coordinates
            xi, zi = find_closest(point[0], point[2], x, z)
            weights[zi, xi] += point[3]  # add weigth to grid
        
        norm = LogNorm(vmin=1e-3, vmax=weights.max())
        plt.figure(num=2)
        plt.pcolormesh(x, z, weights, cmap='magma', norm=norm, shading='auto')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.gca().invert_yaxis()  # z axis is directed downwards
        plt.colorbar()
        plt.tight_layout()
    

def phaseFunPlot(phase, N=1e6):
    """
    Debug function, simulates and plots angular distribution of Phase Function

    Parameters
    ----------
    p : PHASEFUNCTION Object
        Instance of PhaseFunction class
    N : INT, optional
        Number of events to simulate. The default is 1e6.

    Returns
    -------
    None.
    """
    theta = []
    phi = []
    for i in range(N):
        t,p = phase.getAngles()
        theta.append(t)
        phi.append(p)
    
    theta_hist = np.histogram(theta, bins=180//2, range=(0, np.pi))
    phi_hist = np.histogram(phi, bins=360//2, range=(0, 2*np.pi))  # count angle occurrences
    
    # import pdb; pdb.set_trace()
    if plt.fignum_exists(696):  # For multiple plots
        fig = plt.gcf()
        ax = fig.axes
    else:
        fig,ax = plt.subplots(nrows=1, ncols=2, num=696, figsize=(10,5), subplot_kw={'projection': 'polar'})
    ax[0].plot(theta_hist[1][:-1], theta_hist[0], label='g={}'.format(phase.g))
    ax[0].set_title('Theta')
    ax[0].legend(loc='lower center', bbox_to_anchor=(0,-0.2))
    ax[0].set_rscale('symlog')
    ax[0].grid(True, linestyle=':')
    ax[1].plot(phi_hist[1][:-1], phi_hist[0], label='g={}'.format(phase.g))
    ax[1].set_title('Phi')
    ax[1].legend(loc='lower center', bbox_to_anchor=(0,-0.2))
    # ax[1].set_yscale('log')
    ax[1].grid(True, linestyle=':')
    plt.tight_layout()
    
    # return (theta, theta_hist)

# # DEBUG stuff here
# def hg(g,theta):
#         p = ((1-g[:,np.newaxis]**2)/(1+g[:,np.newaxis]**2-2*g[:,np.newaxis]*np.cos(theta))**(3/2))/(4*np.pi)
#         return p

if __name__ == '__main__':
    g = np.array([0., .25, .5, .75, .95])
    # g = np.array([0, 0.5, 0.8])
    phase = [HG(g=x) for x in g]  # list of p
    
    for p in phase:
        phaseFunPlot(p, N=int(1e7))
    
    # alpha = np.arange(360)/180*np.pi
    # ph = hg(g, alpha)
    # plt.figure(2)
    # plt.polar(alpha, ph.T/np.max(ph.T,axis=0))
    # plt.gca().set_rscale('symlog')
    
    
    