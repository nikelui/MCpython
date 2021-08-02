# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:45:18 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Classes for graphical representation of MC simulation (Tissues geometry, photon paths...)
and post-processing data visualization
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import MClayers

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
        plt.figure(1, figsize=(10,8))
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
            ax.plot(photonPath[0], photonPath[1], linewidth=linewidth, linestyle='-', color='yellow')
            
    def showAbsorbed(self, absorbed, xlim=[-10,10], zlim=[-10,10]):
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
        # Create meshgrid to interpolate. Resolution is 0.1mm
        x = np.linspace(xlim[0], xlim[1], num=(xlim[1]-xlim[0])*10+1, dtype=float)
        z = np.linspace(zlim[0], zlim[1], num=(zlim[1]-zlim[0])*10+1, dtype=float)
        weights = np.ones((len(z), len(x)), dtype=float)*1e-16  # weights grid. Slightly larger than 0 for logNorm
        for point in absorbed:
            # find the closest coordinates
            xi, zi = find_closest(point[0], point[2], x, z)
            weights[zi, xi] += point[3]  # add weigth to grid
        
        norm = LogNorm(vmin=1e-3, vmax=weights.max())
        plt.pcolormesh(x, z, weights, cmap='magma', norm=norm, shading='auto')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.tight_layout()
    
    
if __name__ == '__main__':
    g = Geometries()
    g.showGeometry()