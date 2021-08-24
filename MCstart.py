# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:30:16 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Main script to run for iterating
"""
import numpy as np
from datetime import datetime
import pickle, os
from MCphoton import Photon
from MClayers import Slab, Infinite  #TODO: make imports dynamic
from MCPhaseFun import HenyeyGreenstein as HG
from MCgui import Geometries

# TODO: read parameters and tissues from file
param = {}
param['photons_launched'] = 5e4
param['photons_detected'] = 0
param['weigth_threshold'] = 0.1
param['roulette_weigth'] = 10
param['n_sim'] = 10  # number of simulations (to perform statistics)
param['save_path'] = './model3'

# Define tissue list
## model 1
tissues = [
    Infinite(n=1, mua=0., mus=0., order=0, num=0, color='cyan', detect='impinge'),
    Slab(n=1, mua=1, mus=9, order=1, num=1, phase=HG(g=.75), top=0, thick=0.2, color='orange'),
    ]
## model 2
# tissues = [
#     Infinite(n=1, mua=0., mus=0., order=0, num=0, color='cyan', detect='impinge'),
#     Slab(n=1.5, mua=1, mus=9, order=1, num=1, phase=HG(g=0), top=0, thick=100, color='orange'),
#     ]
## model 3
# tissues = [
#     Infinite(n=1, mua=0., mus=0., order=0, num=0, color='cyan', detect='impinge'),
#     Slab(n=1.37, mua=0.1, mus=10, order=1, num=1, phase=HG(g=0.9), top=0, thick=1, color='orange'),
#     Slab(n=1.37, mua=0.1, mus=1, order=2, num=2, phase=HG(g=0), top=1, thick=1, color='chocolate'),
#     Slab(n=1.37, mua=0.2, mus=1, order=3, num=3, phase=HG(g=0.7), top=2, thick=2, color='sienna'),
#     ]
tissues.sort(key=lambda x: x.order, reverse=True)  # sort high to low

# Loop here for multiple simulations
for _i in range(param['n_sim']):
    # results
    detected = []  # store whole photons
    absorbed = []  # store coordinates + dw
    # debug = []
    
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
        ph.angles.append(ph.direction.copy())  # starting direction
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
                    ph.angles.append(ph.direction.copy())
                    ph.scatters += 1
                
            else:  # crossed boundary
                ds = step - dist  # save remaining step
                ph.coordinates += dist*ph.direction  # move photon to boundary
                ph.path.append(ph.coordinates.copy())  # add to path
                while ds > 0:
                    mode = ph.fresnel(current_layer, incident_layer)  # transmit or reflect and update direction
                    ph.angles.append(ph.direction.copy())  # add to angles
                    if mode == 'reflect':
                        incident_layer = None
                    elif mode == 'transmit':
                        if incident_layer.detect:
                            ph.dead = True
                            p_d += 1
                            ph.detected = True  # photon is detected
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
        detected.append(ph)  # new approach: save all emitted photons
    stop = datetime.now()
    print('Simulation {} of {}. Elapsed time: {}'.format(_i+1, param['n_sim'],str(stop-start)))
    # Save data
    # TEST with PICKLE
    if True:
        if not os.path.exists(param['save_path']):
            os.makedirs(param['save_path'])
        with open('{}/data{}.pkl'.format(param['save_path'], _i+1), 'wb') as out_file:
            pickle.dump(detected, out_file)
        # with open('test_data.pkl', 'rb') as in_file:
        #     test_load = pickle.load(in_file)

gg = Geometries(tissues)
ax = gg.showGeometry(xlim=[-.5, .5], zlim=[-.5,.5])
# gg.showPaths(ax, detected, N=1000, linewidth=0.5)
# gg.animatePath(ax, detected, N=10000, M=100, linewidth=0.1)

# asd = gg.showAbsorbed(absorbed, xlim=[-3,3], zlim=[-3,3], res=0.01)
