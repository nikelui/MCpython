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
param['photons_launched'] = 2e5
param['photons_detected'] = 0
param['weigth_threshold'] = 0.1
param['roulette_weigth'] = 10

# Define tissue list
tissues = [
    Infinite(n=1, mua=0., mus=0., order=0, num=0, color='cyan', detect='impinge'),
    Slab(n=1.4, mua=0.2, mus=5, order=1, num=1, phase=HG(g=0.8), top=0, thick=100, color='orange'),
    Slab(n=1.43, mua=0.5, mus=10, order=2, num=2, phase=HG(g=0.8), top=0, thick=0.2, color='chocolate'),
    Slab(n=1.45, mua=1, mus=10, order=3, num=3, phase=HG(g=0.8), top=0, thick=0.1, color='sienna'),
    Slab(n=1.47, mua=1.5, mus=10, order=4, num=4, phase=HG(g=0.8), top=0, thick=0.05, color='darkred')
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
    p_l += 1  # increase photon counter
    ph = Photon()
    ph.path.append(ph.coordinates.copy())  # starting position
    # Detect initial layer and outer layer
    ph.coordinates -= np.array([0, 0, 1e-3])  # move slightly up    
    current_layer_id = ph.find_layer(tissues)
    ph.coordinates += np.array([0, 0, 1e-3])  # restore initial coordinates
    incident_layer_id = ph.find_layer(tissues)
    
    current_layer = tissues[current_layer_id]
    incident_layer = tissues[incident_layer_id]
    
    ph.specular(current_layer, incident_layer)  # specular reflection
    current_layer = incident_layer  # move inside first layer
    # ph.layer = current_layer.number  # to keep track of layer. Is it really needed?
    incident_layer = None
    
    while not ph.dead:  # Main loop: iterate until photon is dead
        step = ph.step(current_layer)
        ph.step_size = step
        # check if photons hits any boundary
        for tissue in tissues:
            dist = tissue.distance(ph)
            if dist < step:
                # need to find new tissue here
                ph.coordinates += (dist + 1e-3)*ph.direction  # move photon slightly in new tissue
                incident_layer_id = ph.find_layer(tissues)
                ph.coordinates -= (dist + 1e-3)*ph.direction  # restore coordinates
                incident_layer = tissues[incident_layer_id]
                break  # after finding current layer, break loop
                
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
            while ds > 0:
                mode = ph.fresnel(current_layer, incident_layer)  # transmit or reflect and update direction                
                if mode == 'reflect':
                    incident_layer = None
                elif mode == 'transmit':
                    if incident_layer.detect:
                        ph.dead = True
                        p_d += 1
                        detected.append(ph)  # save whole photon
                        incident_layer = None
                        break  # do not move further                             
                    else:
                        ds *= (current_layer.mua + current_layer.mus) / (incident_layer.mua + incident_layer.mus)  # update step
                        current_layer = incident_layer
                        incident_layer = None
                # calculate new distance
                for tissue in tissues:
                    dist = tissue.distance(ph)
                    if 1e-4 < dist < ds:
                        # need to find new tissue here
                        ph.coordinates += (dist + 1e-3)*ph.direction  # move photon slightly in new tissue
                        incident_layer_id = ph.find_layer(tissues)
                        ph.coordinates -= (dist + 1e-3)*ph.direction  # restore coordinates
                        incident_layer = tissues[incident_layer_id]
                        break  # after finding current layer, break loop
                if incident_layer is not None:  # if hits another boundary
                    ph.coordinates += dist*ph.direction  # move photon
                    ph.path.append(ph.coordinates.copy())  # add to path
                    ds -= dist  # subtract from remaining step
                    # loop back
                else:  # if does not hit boundary
                    ph.coordinates += ds*ph.direction  # move photon
                    ph.path.append(ph.coordinates.copy())  # add to path
                    ds = 0  # this will break the loop
    debug.append(ph)

stop = datetime.now()

print('Elapsed time: {}'.format(str(stop-start)))

gg = Geometries(tissues)
ax = gg.showGeometry(xlim=[-3, 3], zlim=[-3,3])
# gg.showPaths(ax, detected, N=500, linewidth=0.5)
# gg.animatePath(ax, detected, N=500, M=50)

asd = gg.showAbsorbed(absorbed, xlim=[-3,3], zlim=[-3,3], res=0.01)
