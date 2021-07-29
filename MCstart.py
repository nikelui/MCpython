# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:30:16 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Main script to run for iterating
"""
import numpy as np
from matplotlib import pyplot as plt
from MCphoton import Photon
from MClayers import Slab, Infinite  #TODO: make imports dynamic
from MCPhaseFun import HenyeyGreenstein as HG

# TODO: read parameters and tissues from file
param = {}
param['photons_launched'] = 10
param['photons_detected'] = 0
param['weigth_threshold'] = 0.1
param['roulette_weigth'] = 10

# Define tissue list
tissues = [
    Infinite(order=0, num=0, color='cyan', detect='impinge'),
    Slab(n=1.4, mua=0.05, mus=5, order=1, num=1, phase=HG(g=0.8), top=0, thick=100, color='orange')
    ]
tissues.sort(key=lambda x: x.order, reverse=True)  # sort high to low

# results
detected = []  # store whole photons
absorbed = []  # store coordinates + dw
debug = []

# Some counters
p_l = 0
p_d = 0

if param['photons_detected'] == 0:
   param['photons_detected'] = np.inf  # if zero, continue launching photons until stopped
if param['photons_launched'] == 0:
   param['photons_launched'] = np.inf
   
while p_l < param['photons_launched'] and p_d < param['photons_detected']:
    # First step
    current_layer = tissues[0]
    incident_layer = tissues[1]
    p_l += 1  # increase photon counter
    ph = Photon()
    ph.path.append(ph.coordinates)  # starting position
    ph.specular(current_layer, incident_layer)  # specular reflection
    current_layer = incident_layer  # move inside first layer
    ph.layer = current_layer.number  # to keep track of layer. Is it really needed?
    incident_layer = None
    
    while not ph.dead:  # Main loop: iterate until photon is dead
        step = ph.step(current_layer)
        ph.step_size = step
        # check if photons hits any boundary
        for tissue in tissues:
            dist = tissue.distance(ph)
            if dist < step:
                incident_layer = tissue
                break  # stop at the first crossed layer
        
        if incident_layer is None:  # Not crossed boundary
            ph.coordinates += step*ph.direction  # move photon
            ph.path.append(ph.coordinates)  # add to path
            dw = ph.absorb(current_layer)  # update weigth
            absorbed.append(np.concatenate([ph.coordinates,[dw]]))  # save absorbed
            
            if ph.weigth < param['weigth_threshold']:
                ph.roulette(param['roulette_weigth'])  # kill the photon or increase weigth
            else:
                ph.scatter(current_layer)  # change direction
            
        else:  # crossed boundary
            ds = step - dist  # save remaining step
            ph.coordinates = incident_layer.intersect(ph)  # move photon to boundary
            ph.path.append(ph.coordinates)  # add to path
            mode = ph.fresnel(current_layer, incident_layer)  # transmit or reflect and update direction
            if mode == 'reflect':
                incident_layer = None
                ph.coordinates += ds*ph.direction
                ph.path.append(ph.coordinates)  # add to path
            elif mode == 'transmit':
                if incident_layer.detect:  #TODO: implement detection of absorbed photons
                    ph.dead = True
                    p_d += 1
                    detected.append(ph)  # save whole photon
                    
                else:    
                    ds *= (current_layer.mua + current_layer.mus) / (incident_layer.mua + incident_layer.mus)  # update step                
                    current_layer = incident_layer
                    incident_layer = None
                    ph.coordinates += ds*ph.direction
    debug.append(ph)
    