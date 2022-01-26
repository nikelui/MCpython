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
import addcopyfighandler

# TODO: read parameters and tissues from file
param = {}
param['photons_launched'] = 1e2
param['photons_detected'] = 0
param['weigth_threshold'] = 0.1
param['roulette_weigth'] = 10
param['n_sim'] = 10  # number of simulations (to perform statistics)
param['save_path'] = './whitemc2'
param['debug'] = False
param['lowmem'] = True  # check for low-memory simulations (discard photon.path and photon.angles)
                        # note: this will break the GUI functions to show photon paths
np.set_printoptions(precision=3)
# Define tissue list
## model 1
# tissues = [
#     Infinite(n=1, mua=0., mus=0., order=0, num=0, color='cyan', detect='impinge'),
#     Slab(n=1, mua=1, mus=9, order=1, num=1, phase=HG(g=.75), top=0, thick=0.2, color='orange'),
#     ]
## model 2
# tissues = [
#     Infinite(n=1, mua=0., mus=0., order=0, num=0, color='cyan', detect='impinge'),
#     Slab(n=1.5, mua=1, mus=9, order=1, num=1, phase=HG(g=0), top=0, thick=100, color='orange'),
#     ]
## model 3
# tissues = [
#     Infinite(n=1, mua=0., mus=0., order=0, num=0, color='darkgray', detect='impinge'),
#     Slab(n=1.37, mua=0.1, mus=10, order=3, num=3, phase=HG(g=0.9), top=0, thick=1, color='aquamarine'),
#     Slab(n=1.37, mua=0.1, mus=1, order=2, num=2, phase=HG(g=0), top=1, thick=1, color='turquoise'),
#     Slab(n=1.37, mua=0.2, mus=1, order=1, num=1, phase=HG(g=0.7), top=2, thick=2, color='lightseagreen'),
#     ]
## whiteMC
tissues = [
    Infinite(n=1, mua=0., mus=0., order=0, num=0, color='cyan', detect='impinge'),
    Slab(n=1.4, mua=0, mus=10, order=1, num=1, phase=HG(g=0.8), top=0, thick=1e6, color='orange'),
    ]
tissues.sort(key=lambda x: x.order, reverse=True)  # sort high to low

times = []  # DEBUG

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
        if param['debug']:
            if p_l <= 2:
                print('### PHOTON {} ###'.format(p_l+1))  # DEBUG
        ph = Photon()
        # Detect initial layer and outer layer
        ph.coordinates -= np.array([0, 0, 1e-3])  # move slightly up    
        current_layer_id = ph.find_layer(tissues)
        ph.coordinates += np.array([0, 0, 1e-3])  # restore initial coordinates
        incident_layer_id = ph.find_layer(tissues)
        current_layer = tissues[current_layer_id]
        incident_layer = tissues[incident_layer_id]
        
        if not param['lowmem']:
            ph.path.append(ph.coordinates.copy())  # starting position
            ph.angles.append(ph.direction.copy())  # starting direction
            ph.layers.append(current_layer.number)  # DEBUG, save current layer
        
        ph.spec = ph.specular(current_layer, incident_layer)  # specular reflection
        current_layer = incident_layer  # move inside first layer
        incident_layer = None
        
        while not ph.dead:  # Main loop: iterate until photon is dead
            if ph.step_size == 0:
                ph.step_size = ph.step(current_layer)  # new random step size
            # check if photons hits any boundary
            dists = [t.distance(ph) if t.distance(ph) > 1e-10 else np.inf for t in tissues]
            dist = min(dists)
            # for tissue in tissues:
            #     dist = tissue.distance(ph)
            if dist < ph.step_size:
                # need to find new tissue here
                ph.coordinates += (dist + 1e-3)*ph.direction  # move photon slightly in new tissue
                incident_layer_id = ph.find_layer(tissues)
                # DEBUG
                # if incident_layer_id == 2:
                #     import pdb; pdb.set_trace()
                ph.coordinates -= (dist + 1e-3)*ph.direction  # restore coordinates
                incident_layer = tissues[incident_layer_id]
            
            if incident_layer is None:  # Not crossed boundary
                ph.coordinates += ph.step_size*ph.direction  # move photon
                # DEBUG
                # if ph.coordinates[-1] < -1e-10 or ph.coordinates[-1] > 4+1e-10:  # if escapes in air
                #     import pdb; pdb.set_trace()
                #     ph.find_layer(tissues)
                
                ph.pathlength += ph.step_size  # increase pathlength
                ph.step_size = 0  # reset step
                # DEBUG
                if param['debug']:
                    if p_l <= 2:
                        print('coord: {}, dir: {}, w: {:.3f}, layer: {}'.format(
                            ph.coordinates, ph.direction, ph.weigth, current_layer.number))  # DEBUG
                if not param['lowmem']:
                    ph.path.append(ph.coordinates.copy())  # add to path
                    ph.layers.append(current_layer.number)  # DEBUG, save current layer
                dw = ph.absorb(current_layer)  # update weigth
                if dw > 1e-6:  # only save if absorption is happening
                    absorbed.append(np.concatenate([ph.coordinates.copy(),[dw]]))  # save absorbed
                
                if ph.weigth < param['weigth_threshold']:
                    ph.roulette(param['roulette_weigth'])  # kill the photon or increase weigth
                else:
                    ph.scatter(current_layer)  # change direction
                    if not param['lowmem']:
                        ph.angles.append(ph.direction.copy())
                    ph.scatters += 1
                
            else:  # Crossed boundary
                ph.step_size -= dist  # reduce step
                ph.coordinates += dist*ph.direction  # move photon to boundary
                # DEBUG
                if ph.coordinates[-1] < -1e-10 or ph.coordinates[-1] > 4+1e-10:  # if escapes in air
                    # import pdb; pdb.set_trace()
                    ph.find_layer(tissues)
                ph.pathlength += dist
                if param['debug']:
                    if p_l <= 2:
                        print('coord: {}, dir: {}, w: {:.3f}, layer: {}'.format(
                            ph.coordinates, ph.direction, ph.weigth, current_layer.number))  # DEBUG
                if not param['lowmem']:
                    ph.path.append(ph.coordinates.copy())  # add to path
                    ph.layers.append(current_layer.number)  # DEBUG, save current layer
                # DEBUG
                # if incident_layer.detect:
                #     import pdb; pdb.set_trace()
                mode = ph.fresnel(current_layer, incident_layer)  # transmit or reflect and update direction
                if not param['lowmem']:
                    ph.angles.append(ph.direction.copy())  # add to angles
                
                # dw = ph.absorb(current_layer)  # update weigth
                # if dw > 1e-6:  # only save if absorption is happening
                #     absorbed.append(np.concatenate([ph.coordinates.copy(),[dw]]))  # save absorbed
                
                if ph.weigth < param['weigth_threshold']:
                    ph.roulette(param['roulette_weigth'])  # kill the photon or increase weigth
                
                # reflect or refract
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
                        ph.step_size *= (current_layer.mua + current_layer.mus) /  \
                                        (incident_layer.mua + incident_layer.mus)  # update step
                        current_layer = incident_layer
                        incident_layer = None
                    
                    # # calculate new distance
                    # # for tissue in tissues:
                    # #     dist = tissue.distance(ph)
                    # dists = [t.distance(ph) if t.distance(ph) > 1e-4 else np.inf for t in tissues]
                    # dist = min(dists)
                    # if 1e-4 < dist < ph.step_size:
                    #     # need to find new tissue here
                    #     ph.coordinates += (dist + 1e-3)*ph.direction  # move photon slightly in new tissue
                    #     incident_layer_id = ph.find_layer(tissues)
                    #     ph.coordinates -= (dist + 1e-3)*ph.direction  # restore coordinates
                    #     incident_layer = tissues[incident_layer_id]
                        
                    # if incident_layer is not None:  # if hits another boundary
                    #     ph.coordinates += dist*ph.direction  # move photon
                    #     ph.pathlength += dist
                    #     if param['debug']:
                    #         if p_l <= 2:
                    #             print('coord: {}, dir: {}, w: {:.3f}, layer: {}'.format(
                    #                 ph.coordinates, ph.direction, ph.weigth, current_layer.number))  # DEBUG
                    #     ph.path.append(ph.coordinates.copy())  # add to path
                    #     ds -= dist  # subtract from remaining step
                    #     # loop back
                    # else:  # if does not hit boundary
                    #     ph.coordinates += ds*ph.direction  # move photon
                    #     ph.pathlength += ds
                    #     if param['debug']:
                    #         if p_l <= 2:
                    #             print('coord: {}, dir: {}, w: {:.3f}, layer: {}'.format(
                    #                 ph.coordinates, ph.direction, ph.weigth, current_layer.number))  # DEBUG
                    #     ph.path.append(ph.coordinates.copy())  # add to path
                    #     ds = 0  # this will break the loop
                    
        detected.append(ph)  # new approach: save all emitted photons
    stop = datetime.now()
    print('Simulation {} of {}. Elapsed time: {}'.format(_i+1, param['n_sim'],str(stop-start)))
    times.append((stop-start).total_seconds())
    # Save data
    # TEST with PICKLE
    if True:
        if not os.path.exists(param['save_path']):
            os.makedirs(param['save_path'])
        with open('{}/data{}.pkl'.format(param['save_path'], _i+1), 'wb') as out_file:
            pickle.dump(detected, out_file)
        # with open('test_data.pkl', 'rb') as in_file:
        #     test_load = pickle.load(in_file)
print(r'Average time: {:.2f}+/-{:.2f} s'.format(np.mean(times), np.std(times)))

gg = Geometries(tissues)
ax = gg.showGeometry(xlim=[-100, 100], zlim=[-.5, 100])
gg.showPaths(ax, detected, N=500, linewidth=0.1, alpha=1)
# gg.animatePath(ax[0], detected, N=100, M=50, linewidth=0.5)
# gg.paths_3d(detected, N=1000, xlim=[-2,2], ylim=[-2,2], zlim=[-1,2], alpha=0.3)
# asd = gg.showAbsorbed(absorbed, xlim=[-5,5], zlim=[-3,5], res=0.1)

#%%
## DEBUG
# from matplotlib import pyplot as plt
# plt.figure(45425)
# ph = detected[np.random.randint(0,len(detected))]
# idx = [0, 4, 2, 1]
# l = [idx[n] for n in ph.layers]
# z = [x[-1] for x in ph.path]

# plt.plot(l, '*', zorder=50)
# plt.grid(True, linestyle=':')
# plt.plot(z, 's', zorder=10)
