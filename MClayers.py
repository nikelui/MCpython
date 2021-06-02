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
    def __init__(self, n=1.4, mua=0, mus=1, order=0, **kwargs):
        """Defaults to white Monte Carlo with n = 1.4"""
        self.n = n  # refractive index
        self.mua = mua  # absorption coefficient
        self.mus = mus  # scattering coefficient
        self.order = order  # priority of the layer (0 = lowest)
        self.color = kwargs.get('color', 'white')  # Used for graphical representation
        self.detect = kwargs.get('detect', '')  # Used to define detector layers
    
    def is_inside(self, coord):
        """Determine if a point coord = (x,y,z) is inside the geometry or not"""
        pass
    
    def find_boundary(self, p1, p2):
        """Find the boundary point between two points of coordinates p1, p2 (if exists)"""
        pass
    
    
class Infinite(Layer):
    """Special infinite geometry for surrounding tissue"""
    def __init__(self, n=1, mua=0, mus=0):
        """Defaults to Air layer"""
        super().__init__(n, mua, mus)
        self.type = 'infinite'
    
    def is_inside(self, coord):
        return True  # Always true, unless coord = (x,y,z) is in another layer

    
class Slab(Layer):
    """Horizontal slab of tissue, laying in the xy plane"""
    def __init__(self):
        super().__init__()
        self.type = 'slab'
        self.thickness = 1  # thickness in mm
        self.pos = 0  # position of the top of the layer in the z axis
    
    def is_inside(self, coord):
        """Determine if a point coord = (x,y,z) is inside the geometry or not"""
        x,y,z = coord  # unpack coordinates
        if (self.pos + self.thickness) <= z <= self.pos:  # only need to check z
            return True
        else:
            return False
    
    def find_boundary(self, p1, p2):
        """Find the boundary point between two points of coordinates p1, p2 (if exists)"""
        p1 = np.array(p1)  # to use vector operation
        p2 = np.array(p2)
        # First some checks
        if p1[-1] > p2[-1]:
            if not(p2[-1] <= self.top <= p1[-1] or p2[-1] <= (self.top + self.thickness) <= p1[-1]):
                return None
        elif p2[-1] > p1[-1]:
            if not(p1[-1] <= self.top <= p2[-1] or p1[-1] <= (self.top + self.thickness) <= p2[-1]):
                return None
                
        v = p2 - p1  # direction vector
        
        
        