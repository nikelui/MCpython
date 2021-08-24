# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:12:29 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script to post-process the simulated data
"""
import numpy as np
import pickle, os
from datetime import datetime
from MCphoton import Photon
from matplotlib import pyplot as plt

path = './model1'

# Total reflectance / transmittance
tot_refl = []
tot_trans = []

# Radial distribution
radial_refl = []
radial_trans = []
r_step = 0.1  # mm
r_detect = np.arange(0, 10, r_step)  # radial detector

# Angular distribution
angular_refl = []
angular_trans = []
a_step = np.pi/90  # radians
a_detect = np.arange(0, np.pi/2, a_step)  # angular detector

# measure elapsed time
start = datetime.now()

for _i, file in enumerate(os.listdir(path)):
    print('File {} of {}'.format(_i+1, len(os.listdir(path))))
    with open('{}/{}'.format(path, file), 'rb') as f_in:
        dataset = pickle.load(f_in)
        n_emitted = len(dataset)  # photons emitted
        tot_r = 0    # reflected counter
        tot_t = 0  # transmitted counter
        radial_r = np.zeros(len(r_detect))
        radial_t = np.zeros(len(r_detect))
        angular_r = np.zeros(len(a_detect))
        angular_t = np.zeros(len(a_detect))
        for photon in dataset:
            if photon.detected:  # only process if detected
                if photon.coordinates[2] < 1e-8:  # allow tolerance
                    tot_r += photon.weigth  # reflected to the surface
                    # find radial position
                    rho = np.sqrt(photon.coordinates[0]**2 + photon.coordinates[1]**2)  # radius
                    idx = max(np.searchsorted(r_detect, rho) - 1, 0)  # need -1 to get correct position
                    radial_r[idx] += photon.weigth
                    # find angular position
                    theta = np.arccos(np.abs(photon.direction[2]))
                    idx = max(np.searchsorted(a_detect, theta) - 1, 0)  # need -1 to get correct position
                    angular_r[idx] += photon.weigth
                    
                # TODO: better check on coordinates
                else:  # this is not very rigorous, but for now it will do
                    tot_t += photon.weigth  # transmitted to bottom
                    # find radial position
                    rho = np.sqrt(photon.coordinates[0]**2 + photon.coordinates[1]**2)  # radius
                    idx = max(np.searchsorted(r_detect, rho) - 1, 0)  # need -1 to get correct position
                    radial_t[idx] += photon.weigth
                    # find angular position
                    theta = np.arccos(np.abs(photon.direction[2]))
                    idx = max(np.searchsorted(a_detect, theta) - 1, 0)  # need -1 to get correct position
                    angular_t[idx] += photon.weigth
        # append to arrays (for statistics)
        tot_refl.append(tot_r)
        tot_trans.append(tot_t)
        radial_refl.append(radial_r)
        radial_trans.append(radial_t)
        angular_refl.append(angular_r)
        angular_trans.append(angular_t)
        
stop = datetime.now()
print('Elapsed time: {}'.format(str(stop-start)))

print('Reflectance = {:.2f}%  std:{:.2f}%'.format(np.mean(tot_refl)/n_emitted *100,
                                                 np.std(tot_refl)/n_emitted *100))
print('Transmittance = {:.2f}%  std:{:.2f}%'.format(np.mean(tot_trans)/n_emitted *100,
                                                 np.std(tot_trans)/n_emitted *100))
# radial distribution
fig, ax = plt.subplots(nrows=1, ncols=2, num=1, figsize=(10,4))
# ax[0].errorbar(r_detect + r_step, np.mean(radial_refl, axis=0)/n_emitted,
#                yerr=np.std(radial_refl, axis=0)/n_emitted)
ax[0].plot(r_detect + r_step, np.mean(radial_refl, axis=0)/n_emitted, 'x')
ax[0].set_title('Reflectance')
ax[0].set_xlabel('mm')
ax[0].set_xlim([0, 2])
ax[0].set_yscale('log')
ax[0].grid(True, linestyle=':')
# ax[1].errorbar(r_detect + r_step, np.mean(radial_trans, axis=0)/n_emitted,
#                yerr=np.std(radial_trans, axis=0)/n_emitted)
ax[1].plot(r_detect + r_step, np.mean(radial_trans, axis=0)/n_emitted, 'x')
ax[1].set_title('Transmittance')
ax[1].set_xlabel('mm')
ax[1].set_xlim([0, 2])
ax[1].set_yscale('log')
ax[1].grid(True, linestyle=':')
plt.tight_layout()

# angular distribution
fig, ax = plt.subplots(nrows=1, ncols=2, num=2, figsize=(10,4))
# ax[0].errorbar(a_detect + a_step, np.mean(angular_refl, axis=0)/n_emitted,
#                yerr=np.std(angular_refl, axis=0)/n_emitted)
ax[0].plot(a_detect + a_step, np.mean(angular_refl, axis=0)/n_emitted, 'x')
ax[0].set_title('Reflectance')
ax[0].set_xlabel('[rad]')
ax[0].grid(True, linestyle=':')
# ax[1].errorbar(a_detect + a_step, np.mean(angular_trans, axis=0)/n_emitted,
#                yerr=np.std(angular_trans, axis=0)/n_emitted)
ax[1].plot(a_detect + a_step, np.mean(angular_trans, axis=0)/n_emitted, 'x')
ax[1].set_title('Transmittance')
ax[1].set_xlabel('[rad]')
ax[1].grid(True, linestyle=':')
plt.tight_layout()