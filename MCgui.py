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
from matplotlib.patches import Rectangle  #TODO: add other geometries
import MClayers

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
    
    def showPaths(self, ax, photons, **kwargs):
        """
        Plots the geometry and shows photon paths

        Parameters
        ----------
        ax : MATPLOTLIB AXIS OBJECT
        
        photons : LIST of simulated PHOTON OBJECTS
            Results of the simulation, with scattering paths
            
        Returns
        -------
        None.
        """
        linewidth = kwargs.get('linewidth', 1)
        for photon in photons:
            photonPath = np.array([[x[0] for x in photon.path], [x[2] for x in photon.path]])
            ax.plot(photonPath[0], photonPath[1], linewidth=linewidth, linestyle='-', color='yellow')
        
    
    
if __name__ == '__main__':
    g = Geometries()
    g.showGeometry()