# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:47:12 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Class to represent photon packets
"""
import numpy as np
from operator import attrgetter  # used to find maximum attribute

class Photon:
    """Class to represent photon packets"""
    def __init__(self, **kwargs):
        self.coordinates = np.array([0,0,0], dtype=float)  # in mm. Default to axis origin
        self.direction = np.array([0,0,1], dtype=float)  # default to positive z direction
        self.weigth = 1.0  # photon packet weigth
        self.spec = 0  # specular reflection at the surface
        self.dead = False  # flag to check photon termination
        self.detected = False  # Flag to determine if a photon is detected
        self.step_size = 0  # step size (in mm). Update at every iteration
        self.scatters = 0  # number of scattering events
        self.path = []  # to store photon path
        self.angles = []  # to store scattering directions
    
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
        step = -np.log(np.random.rand()) / (tissue.mua + tissue.mus)
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
        mux, muy, muz = self.direction.copy()  # unpack components for convenience
        cos_theta, phi = tissue.phaseFunction.getAngles()  # randomly sample scattering angles
        # Save some values to optimize (trigonometric functions are expensive)
        temp = np.sqrt(1-muz**2)
        sin_theta = np.sqrt(1 - cos_theta**2)
        cos_phi = np.cos(phi)
        # sin_phi = np.sin(phi)
        if phi <= np.pi:  # new approach, sqrt is less expensive than sin
            sin_phi = np.sqrt(1 - cos_phi**2)
        else:
            sin_phi = -np.sqrt(1 - cos_phi**2)

        if np.abs(muz) < 0.99:
        # OLD approach
            new_dir = np.array([
                sin_theta/temp * (mux*muz*cos_phi - muy*sin_phi) + mux*cos_theta,  # mux'
                sin_theta/temp * (muy*muz*cos_phi + mux*sin_phi) + muy*cos_theta,  # muy'
                -sin_theta*cos_phi*temp + muz*cos_theta                            # muz'
                ])
        else:  # use this to avoid division by zero
            new_dir = np.array([
                sin_theta * cos_phi,    # mux
                sin_theta * sin_phi,    # muy
                cos_theta * np.sign(muz)  # muz
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
        return Rsp  # needs to be added to total reflection
    
    def fresnel(self, tissue1, tissue2):
        """
        Determine if photon is reflected or refracted at the boundary between two tissues
        and update the direction.

        Parameters
        ----------
        tissue1 : LAYER OBJECT
            An instance of the Layer class representing the current tissue layer.
        tissue2 : LAYER OBJECT
            An instance of the Layer class representing the incident tissue layer.

        Returns
        -------
        mode : STRING
            Either 'reflect' or 'transmit'
        """
        top_layer = max((tissue1, tissue2), key=attrgetter('order'))  # find top layer
        ai = top_layer.incident(self)
        # Check for total internal reflection
        if tissue1.n > tissue2.n and ai > np.arcsin(tissue2.n/tissue1.n):
            Ri = 1  # internal reflection
        elif tissue1.n == tissue2.n:  # matched reflection index
            Ri = 0  # transmit with same direction
        else:
            at = np.arcsin(tissue1.n/tissue2.n * np.sin(ai))  # Snell law, transmission angle
            Ri = 0.5*( np.sin(ai-at)**2/np.sin(ai+at)**2 + np.tan(ai-at)**2/np.tan(ai+at)**2 ) 
        # Randomly determine if reflect or transmit
        norm = top_layer.normal(self.coordinates)  # assume the photon is on the boundary
        if np.random.rand() < Ri:  # reflect
            new_dir = -2*norm * self.direction.copy() + self.direction.copy()
            mode = 'reflect'
        else:  # transmit
            k = np.sqrt(1 - tissue1.n**2/tissue2.n**2 * (1 - np.cos(ai)**2)) - tissue1.n/tissue2.n * np.cos(ai)
            new_dir = k*norm + self.direction.copy() * tissue1.n/tissue2.n
            mode = 'transmit'
            
            # new_dir = self.direction  # DEBUG
            # mode = 'transmit'  # DEBUG        
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
            self.weigth = 0  # maybe it is not necessary    
            self.dead = True
        return
    
    def find_layer(self, tissues):
        """
        Find the index of the current layer.
        
        Parameters
        ----------
        tissues : LIST of LAYER objects
            An ordered list that contains the geometry

        Returns
        -------
        idx : INT
            index of the current layer
        """
        for idx, tissue in enumerate(tissues):
            if tissue.is_inside(self.coordinates):
                return idx  # should stop at the first true value