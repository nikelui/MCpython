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
    def __init__(self, **kwargs):
        self.coordinates = np.array([0,0,0], dtype=float)  # in mm. Default to axis origin
        self.direction = np.array([0,0,1], dtype=float)  # default to positive z direction
        self.weigth = 1.0  # photon packet weigth
        self.dead = False  # flag to check photon termination
        self.step_size = 0  # step size (in mm). Update at every iteration
        self.scatters = 0  # number of scattering events
        self.path = []  # to store photon path
        # self.layer = 0  # to keep track of the current tissue layer
    
    def step(self, tissue):
        """
        Generate random step size
        
        Parameters
        ----------
        tissue : LAYER OBJECT
            An instance of the Layer class representing the current tissue layer.

        Returns
        -------
        step : FLOAT
            step size in mm
        """
        step = np.log(np.random.rand()) / (tissue.mua + tissue.mus)
        return step
    
    def absorb(self, tissue):
        """
        Calculate absorbption and decrease photon weigth

        Parameters
        ----------
        tissue : LAYER OBJECT
            An instance of the Layer class representing the current tissue layer.

        Returns
        -------
        dw : FLOAT
            absorbed photon weigth
        """
        w1 = self.weigth * tissue.mus / (tissue.mua + tissue.mus)  # new weigth
        dw = self.weigth - w1  # absorbed weigth
        self.weigth = w1  # update weight
        return dw
    
    def scatter(self, tissue):
        """
        Calculate new direction after scattering

        Parameters
        ----------
        tissue : LAYER OBJECT
            An instance of the Layer class representing the current tissue layer.

        Returns
        -------
        new_dir : FLOAT ARRAY
            new scattering direction
        """
        theta, phi = tissue.phaseFunction.getAngles()  # randomly sample scattering angles
        mux, muy, muz = self.direction  # unpack components for convenience
        
        if np.abs(muz) < 0.9999:
            new_dir = np.array([
                np.sin(theta)/np.sqrt(1-muz**2) * (mux*muz*np.cos(phi) - muy*np.sin(phi)) + mux*np.cos(theta),  # mux'
                np.sin(theta)/np.sqrt(1-muz**2) * (muy*muz*np.cos(phi) - mux*np.sin(phi)) + muy*np.cos(theta),  # muy'
                -np.sin(theta)*np.cos(theta)*np.sqrt(1-muz**2) + muz*np.cos(theta)                              # muz'
                ])
        else:  # use this to avoid division by zero
            new_dir = np.array([
                np.sin(theta)*np.cos(phi),    # mux
                np.sin(theta)*np.sin(phi),    # muy
                np.cos(theta) * np.sign(muz)  # muz
                ])
            
        self.direction = new_dir  # update direction 
        # return new_dir
    
    def specular(self, tissue1, tissue2):
        """
        To be used at the surface, when there is a refracting index mismatch

        Parameters
        ----------
        tissue1 : LAYER OBJECT
            An instance of the Layer class representing the outside tissue layer.
        tissue2 : LAYER OBJECT
            An instance of the Layer class representing the incident tissue layer.

        Returns
        -------
        None.
        """
        Rsp = (tissue1.n - tissue2.n)**2/(tissue1.n + tissue2.n)**2  # specular reflection
        self.weigth -= Rsp  # update weigth
    
    def fresnel(self, tissue1, tissue2):
        """
        Determine if photon is reflected or refracted at the boundary between two tissues
        and update the direction.

        Parameters
        ----------
        tissue1 : LAYER OBJECT
            An instance of the Layer class representing the current tissue layer.
        tissue2 : LAYER OBJECT
            An instance of the Layer class representing the current tissue layer.

        Returns
        -------
        mode : STRING
            Either 'reflect' or 'transmit'
        """
        ai = tissue2.incident(self)
        at = np.arcsin(tissue1.n/tissue2.n * np.sin(ai))  # Snell law, transmission angle
        # Check for total internal reflection
        if tissue1.n > tissue2.n and ai > np.arcsin(tissue2.n/tissue1.n):
            Ri = 1  # internal reflection
        else:
            Ri = 0.5*( np.sin(ai-at)**2/np.sin(ai+at)**2 + np.tan(ai-at)**2/np.tan(ai+at)**2 )
        # Randomly determine if reflect or transmit
        norm = tissue2.normal(self.coordinates)  # assume the photon is on the boundary
        if np.random.rand() <= Ri:  # reflect
            new_dir = -2*norm * self.direction + self.direction
            mode = 'reflect'
        else:  # transmit
            k = np.sqrt(1 - tissue1.n**2/tissue2.n**2 * (1 - np.cos(ai)**2) - tissue1.n/tissue2.n * np.cos(ai))
            new_dir = k*norm + self.direction * tissue1.n/tissue2.n
            mode = 'transmit'
        self.direction = new_dir
        return mode
        
    def roulette(self, r=10):
        """
        Randomly terminate photons if weigth is small enough.

        Parameters
        ----------    
        r : INT
            The photon have 1/r chance to survive the roulette. If it survives,
            the weigth is increased r times.

        Returns
        -------
        None.
        """
        RNG = np.random.randint(1, r+1)
        if RNG == r:  # the photon survives
            self.weigth *= r
        else:  # the photon is terminated
            self.dead = True
        return