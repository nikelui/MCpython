# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:30:16 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Main script to run for iterating
"""
import numpy as np
from datetime import datetime
from MCphoton import Photon
from MClayers import Slab, Infinite  #TODO: make imports dynamic
from MCPhaseFun import HenyeyGreenstein as HG
from MCgui import Geometries

# TODO: read parameters and tissues from file
param = {}
param['photons_launched'] = 1e5
param['photons_detected'] = 0
param['weigth_threshold'] = 0.1
param['roulette_weigth'] = 10

# Define tissue list
tissues = [
    Infinite(n=1, mua=0., mus=0., order=0, num=0, color='cyan', detect='impinge'),
    Slab(n=1.4, mua=0.2, mus=5, order=1, num=1, phase=HG(g=0), top=0, thick=100, color='orange')
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

# measure elapsed time
start = datetime.now()

while p_l < param['photons_launched'] and p_d < param['photons_detected']:
    # First step
    current_layer = tissues[1]  #TODO: define a way to detect outer layer and first layer (e.g. coordinates)
    incident_layer = tissues[0]
    p_l += 1  # increase photon counter
    ph = Photon()
    ph.path.append(ph.coordinates.copy())  # starting position
    ph.specular(current_layer, incident_layer)  # specular reflection
    current_layer = incident_layer  # move inside first layer
    # ph.layer = current_layer.number  # to keep track of layer. Is it really needed?
    incident_layer = None
    
    while not ph.dead:  # Main loop: iterate until photon is dead
        step = ph.step(current_layer)
        ph.step_size = step
        # check if photons hits any boundary
        for tissue in tissues:
            disc = ph.coordinates  # DEBUG
            disd = ph.direction  # DEBUG
            diss = ph.step_size  # DEBUG
            dist = tissue.distance(ph)
            if dist < step:
                # need to find new tissue here
                new_coord = ph.coordinates + step*ph.direction
                for layer in tissues:
                    if layer.is_inside(new_coord):
                        incident_layer = layer
                        break  # exit inner layers loop
                break  # exit outer tissues loop
        
        if incident_layer is None:  # Not crossed boundary
            ph.coordinates += step*ph.direction  # move photon
            ph.path.append(ph.coordinates.copy())  # add to path
            dw = ph.absorb(current_layer)  # update weigth
            if dw > 1e-6:  # only save if absorption is happening
                absorbed.append(np.concatenate([ph.coordinates.copy(),[dw]]))  # save absorbed
            
            if ph.weigth < param['weigth_threshold']:
                ph.roulette(param['roulette_weigth'])  # kill the photon or increase weigth
            else:
                ph.scatter(current_layer)  # change direction
                ph.scatters += 1
            
        else:  # crossed boundary
            ds = step - dist  # save remaining step
            ph.coordinates += dist*ph.direction  # move photon to boundary
            
            ph.path.append(ph.coordinates.copy())  # add to path
            mode = ph.fresnel(current_layer, incident_layer)  # transmit or reflect and update direction
            if mode == 'reflect':
                incident_layer = None
                ph.coordinates += ds*ph.direction
                ph.path.append(ph.coordinates.copy())  # add to path
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

stop = datetime.now()

print('Elapsed time: {}'.format(str(stop-start)))

gg = Geometries(tissues)
ax = gg.showGeometry(xlim=[-2, 2], zlim=[-2, 2])
# gg.showPaths(ax, detected, N=1000, linewidth=0.5)

gg.animatePath(ax, detected, M=50)

asd = gg.showAbsorbed(absorbed)
