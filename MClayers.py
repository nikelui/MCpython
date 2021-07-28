# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:40:25 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Geometry objects used in the Monte Carlo code
"""
import numpy as np

class Layer:
    """Base geometry class, to subclass"""
    def __init__(self, n=1.4, mua=0, mus=1, order=0, num=0, **kwargs):
        """Defaults to white Monte Carlo with n = 1.4"""
        self.n = n  # refractive index
        self.mua = mua  # absorption coefficient
        self.mus = mus  # scattering coefficient
        self.order = order  # priority of the layer (0 = lowest). Should be unique
        self.number = num  # layer identification number. Should be unique
        self.color = kwargs.get('color', 'white')  # Used for graphical representation
        self.detect = kwargs.get('detect', '')  # Used to define detector layers
    
    def is_inside(self, coord):
        """Determine if a point coord = (x,y,z) is inside the geometry or not"""
        pass
    
    def distance(self, photon):
        """
        Calculate the distance between the tissue boundary and the photon

        Parameters
        ----------
        photon : Photon object
            An instance of the Photon class

        Returns
        -------
        dist : FLOAT.
            Distance between the photon and the tissue. It is equal to np.inf if the
            path do not intersect
        """
        pass
    
    def intersect(self, photon):
        """
        Calculate the intersection between the photon path and the tissue boundary.

        Parameters
        ----------
        photon : Photon object
            An instance of the Photon class

        Returns
        -------
        intersect : FLOAT ARRAY.
            Coordinates of the intersection. Returns None if no intersection is found
        """
        pass
    
    
class Infinite(Layer):
    """Special infinite geometry for surrounding tissue"""
    def __init__(self, n=1, mua=0, mus=0):
        """Defaults to Air layer"""
        super().__init__(n, mua, mus)
        self.type = 'infinite'
    
    def is_inside(self, coord):
        return True  # Always true, unless coord = (x,y,z) is in another layer
    
    def distance(self, photon):
        """
        Calculate the distance between the tissue boundary and the photon.

        Parameters
        ----------
        photon : Photon object
            An instance of the Photon class

        Returns
        -------
        dist : FLOAT.
            Distance between the photon and the tissue. It is equal to np.inf if the
            path do not intersect
        """
        return np.inf  # there is no boundary
    
    def intersect(self, photon):
        """
        Calculate the intersection between the photon path and the tissue boundary.

        Parameters
        ----------
        photon : Photon object
            An instance of the Photon class

        Returns
        -------
        intersect : FLOAT ARRAY.
            Coordinates of the intersection. Returns None if no intersection is found
        """
        return None  # there is no boundary
    
    
class Slab(Layer):
    """Horizontal slab of tissue, laying in the xy plane"""
    def __init__(self, top=0, thick=1):
        super().__init__()
        self.type = 'slab'
        self.thickness = thick  # thickness in mm
        self.top = top  # position of the top of the layer in the z axis
    
    def is_inside(self, coord):
        """
        Determine if a point coord = (x,y,z) is inside the geometry or not
        
        Parameters
        ----------
        coord : FLOAT ARRAY
            Coordinates of the point.

        Returns
        -------
        bool
            True if the point is contained in the tissue layer, False otherwise.
        """
        x,y,z = coord  # unpack coordinates
        if (self.top + self.thickness) <= z <= self.top:  # only need to check z
            return True
        else:
            return False
    
    def distance(self, photon):
        """
        Calculate the distance between the tissue boundary and the photon.

        Parameters
        ----------
        photon : Photon object
            An instance of the Photon class

        Returns
        -------
        dist : FLOAT.
            Distance between the photon and the tissue. It is equal to np.inf if the
            path do not intersect
        """
        if photon.direction[2] < 0:  # if directed toward negative z
            dist = (self.top - photon.coordinates[2]) / photon.direction[2]
        elif photon.direction[2] > 0:  # if directed toward positive z
            dist = (self.top + self.thickness - photon.coordinates[2]) / photon.direction[2]
        else:  # if directed horizontally
            dist = np.inf
        return dist
    
    def intersect(self, photon):
        """
        Calculate the intersection between the photon path and the tissue boundary.

        Parameters
        ----------
        photon : Photon object
            An instance of the Photon class

        Returns
        -------
        intersect : FLOAT ARRAY.
            Coordinates of the intersection. Returns None if no intersection is found
        """
        # Sanity check: distance < stepsize
        dist = self.distance(photon)
        if dist < photon.step:
            if photon.direction[2] < 0:  # if directed toward negative z
                intersect = np.array([
                    photon.coordinates[0] + dist * photon.direction[0],  # x coordinate
                    photon.coordinates[1] + dist * photon.direction[1],  # y coordinate
                    self.top])                                           # z coordinate      
            elif photon.direction[2] > 0:  # if directed toward positive z
                intersect = np.array([
                    photon.coordinates[0] + dist * photon.direction[0],  # x coordinate
                    photon.coordinates[1] + dist * photon.direction[1],  # y coordinate
                    self.top + self.thickness])                          # z coordinate
            else:  # if directed horizontally
                intersect = None
        else:
            intersect = None
        return intersect
    
        
        
        
        
        