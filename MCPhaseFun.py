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
        cos_theta, cos_phi : (FLOAT, FLOAT)
            Scattering cosine directors in the longitudinal and azimuthal directions
        """
        phi = 2 * np.pi * np.random.rand()  # uniform probability in the azimuthal direction
        
        g = self.g  # for convenience
        if g > 1e-3:
            cos_theta = (1 + g**2 - ((1-g**2) / (1-g+2*g*np.random.rand()))**2) / (2*g)  # HG
        else:
            cos_theta = 2 * np.random.rand() - 1  # g = 0 -> isotropical scattering
            # cos_theta = np.cos(2 * np.pi * np.random.rand())
        
        return cos_theta, phi


if __name__ == '__main__':
    ph = HenyeyGreenstein(g=0.8)
    phi = np.zeros((10000,1))
    cos_theta = np.zeros((10000,1))
    for _i in range(10000):
        cos_theta[_i], phi[_i] = ph.getAngles()
    