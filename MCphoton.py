# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:47:12 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Class to represent photon packets
"""
import numpy as np

class Photon:
    """Class to represent photon packets"""
    def __init__(self):
        self.coordinates = np.array([0,0,0])  # in mm. Default to axis origin
        self.direction = np.array([0,0,1])  # default to positive z direction
        self.weigth = 1  # photon packet weigth
        self.dead = False  # flag to check photon termination
        self.step = 0  # step size (in mm). Update at every iteration
        self.scatters = 0  # number of scattering events
        self.layer = 0  # to keep track of the current tissue layer
        
    def absorb(self, tissue):
        """
        Calculate absorbption and decrease photon weigth

        Parameters
        ----------
        tissue : LAYER OBJECT
            An instance of the ayer class representing the current tissue layer.

        Returns
        -------
        dw : FLOAT
            absorbed photon weigth
        coord : FLOAT ARRAY
            coordinates where absorption happened.
        """
        w1 = self.weigth * tissue.mus / (tissue.mua + tissue.mus)  # new weigth
        dw = self.weigth - w1  # absorbed weigth
        self.weigth = w1  # update
        
        return dw, self.coordinates
    
    def scatter(self, tissue):
        """
        Calculate new direction after scattering

        Parameters
        ----------
        tissue : LAYER OBJECT
            An instance of the ayer class representing the current tissue layer.

        Returns
        -------
        dir : FLOAT ARRAY
            new scattering direction
        coord : FLOAT ARRAY
            coordinates where scattering happened.
        """
        pass
    
    def roulette(self, threshold=0.1, r=10):
        """
        Randomly terminate photons if weigth is small enough.

        Parameters
        ----------
        threshold : FLOAT
            If the weigth is lower, perform the roulette.         
        r : INT
            The photon have 1/r chance to survive the roulette. If it survives,
            the weigth is increased r times.

        Returns
        -------
        None.
        """
        if self.weigth < threshold:
            RNG = np.random.randint(1, r+1)
            if RNG == r:  # the photon survives
                self.weigth *= r
            else:  # the photon is terminated
                self.dead = True
        return