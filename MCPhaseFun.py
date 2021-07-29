# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 09:24:32 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Collection of phase functions
"""
import numpy as np

class PhaseFunction:
    """Base class for phase functions"""
    def __init__(self):
        #TODO: define phase function parameters (e.g. g1, g2)
        pass
    
    def getAngles(self):
        """
        Returns randomly sampled scattering angles

        Returns
        -------
        theta, phi : (FLOAT, FLOAT)
            Scattering angles in the longitudinal and azimuthal directions
        """
        pass
    
    
class HenyeyGreenstein(PhaseFunction):
    """Base class for phase functions"""
    def __init__(self, g):
        # here go the phase function parameters
        self.g = g
    
    def getAngles(self):
        """
        Returns randomly sampled scattering angles

        Returns
        -------
        theta, phi : (FLOAT, FLOAT)
            Scattering angles in the longitudinal and azimuthal directions
        """
        phi = 2 * np.pi * np.random.rand()  # uniform probability in the azimuthal direction
        
        g = self.g  # for convenience
        if g != 0:
            cos_theta = (1 + g**2 - ((1-g**2) / (1-g+2*g*np.random.rand()))**2) / 2*g  # HG
        else:
            cos_theta = 2*np.random.rand() - 1  # g = 0 -> isotropical scattering
        theta = np.arccos(cos_theta)
        
        return theta, phi
        