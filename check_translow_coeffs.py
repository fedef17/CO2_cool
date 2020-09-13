#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
from matplotlib import cm
#import climtools_lib as ctl

from scipy import io
import scipy.constants as const
import pickle
from scipy.interpolate import PchipInterpolator as spline
import scipy.signal as signal

if os.uname()[1] == 'ff-clevo':
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/SpectRobot/')
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/pythall/')
    cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'
else:
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'
    cart_out_2 = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE/'

import newparam_lib as npl

##############################################################
kbc = const.k/(const.h*100*const.c) # 0.69503
kboltz = 1.38064853e-23 # J/K
E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition

cp = 1.005e7 # specific enthalpy dry air - erg g-1 K-1
#############################################################

allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweigths = dict(zip(allatms, atmweigths))
allco2 = np.arange(1,8)

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v2.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt_v2.p'))

n_alts = 54

all_alts = atm_pt[('mle', 'alts')]
alts = atm_pt[('mle', 'alts')][:n_alts]

pres = atm_pt[('mle', 'pres')]
x = np.log(1000./pres)
n_alts_trlo = np.sum(x < 12.5)
print('low trans at {}'.format(alts[n_alts_trlo]))

all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

n_alts_lte = 40

figs = []
a0s = []
a1s = []
tot_coeff_co2 = pickle.load(open(cart_out_2 + 'tot_coeffs_co2_NLTE.p', 'rb'))


cco2 = 4


from matplotlib.colors import LogNorm
fig = plt.figure()
coef = all_coeffs[('mls', cco2, 'acoeff')]
plt.imshow(np.abs(coef), norm=LogNorm(vmin=0.01, vmax=20000))
fig.savefig(cart_out_2 + 'check_acoeff_LTE.pdf')

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
axes = np.squeeze(np.reshape(axes, (1,6)))
for ii, atm in enumerate(allatms):
    ax = axes[ii]
    coef1 = all_coeffs_nlte[(atm, cco2, 'acoeff')]
    ax.imshow(np.abs(coef1), norm=LogNorm(vmin=0.01, vmax=20000))
    ax.set_title(atm)
fig.savefig(cart_out_2 + 'check_acoeff_NLTE.pdf')

fig = plt.figure()
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
axes = np.squeeze(np.reshape(axes, (1,6)))
for ii, atm in enumerate(allatms):
    ax = axes[ii]
    hr_nlte_fun = all_coeffs_nlte[(atm, cco2, 'hr_nlte_fb')]+all_coeffs_nlte[(atm, cco2, 'hr_nlte_iso')]
    hr_nlte_hot = all_coeffs_nlte[(atm, cco2, 'hr_nlte_hot')]
    hr_nlte = all_coeffs_nlte[(atm, cco2, 'hr_nlte')]
    hr_lte_fun, hr_lte_hot = npl.hr_LTE_FB_vs_ob(atm, cco2)
    hr_lte = all_coeffs_nlte[(atm, cco2, 'hr_lte')]
    hr_ref = all_coeffs[(atm, cco2, 'hr_ref')]
    cols = npl.color_set(5)
    ax.plot(hr_lte_fun, all_alts, label = 'LTE fun', color = cols[0])
    ax.plot(hr_lte_hot, all_alts, label = 'LTE hot', color = cols[2])
    ax.plot(hr_lte, all_alts, label = 'LTE', color = cols[4])
    ax.plot(hr_nlte_fun, all_alts, label = 'NLTE fun', color = cols[0], linestyle = '--')
    ax.plot(hr_nlte_hot, all_alts, label = 'NLTE hot', color = cols[2], linestyle = '--')
    ax.plot(hr_nlte, all_alts, label = 'NLTE', color = cols[4], linestyle = '--')
    ax.plot(hr_ref, all_alts, label = 'LTE ref', color = cols[3])
    ax.set_title(atm)
    ax.set_ylim(20, 90)
    ax.set_xlim(-20, 20)
    ax.grid()

plt.subplots_adjust(bottom = 0.1)
fig.legend(loc = 'lower center')
fig.savefig(cart_out_2 + 'check_HRs_NLTE.pdf')


fig = plt.figure()
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
axes = np.squeeze(np.reshape(axes, (1,6)))
for ii, atm in enumerate(allatms):
    ax = axes[ii]
    hr_nlte_fun = all_coeffs_nlte[(atm, cco2, 'hr_nlte_fb')]+all_coeffs_nlte[(atm, cco2, 'hr_nlte_iso')]
    hr_nlte_hot = all_coeffs_nlte[(atm, cco2, 'hr_nlte_hot')]
    hr_nlte = all_coeffs_nlte[(atm, cco2, 'hr_nlte')]
    hr_lte_fun, hr_lte_hot = npl.hr_LTE_FB_vs_ob(atm, cco2)
    hr_lte = all_coeffs_nlte[(atm, cco2, 'hr_lte')]
    hr_ref = all_coeffs[(atm, cco2, 'hr_ref')]
    cols = npl.color_set(5)

    ratio1 = hr_nlte/hr_lte
    ratio1abs = np.abs(hr_nlte)/np.abs(hr_lte)
    ratio1run = npl.running_mean(ratio1abs, 5, remove_nans = True, keep_length = True)
    ratio2a = hr_nlte_fun/hr_lte_fun
    ratio2b = hr_nlte_hot/hr_lte_hot
    ratio3a = np.abs(hr_nlte_fun)/np.abs(hr_lte_fun)
    ratio3b = np.abs(hr_nlte_hot)/np.abs(hr_lte_hot)

    pio = signal.find_peaks(ratio3a, threshold = 5)
    pio2 = signal.find_peaks(1/ratio3a, threshold = 5)
    for co in np.append([pio[0], pio2[0]]):
        ratio3a[co] = np.mean([ratio3a[co-1], ratio3a[co+1]])
    pio = signal.find_peaks(ratio3b, threshold = 5)
    pio2 = signal.find_peaks(1/ratio3b, threshold = 5)
    for co in np.append([pio[0], pio2[0]]):
        ratio3b[co] = np.mean([ratio3b[co-1], ratio3b[co+1]])

    ratio4a = npl.running_mean(ratio3a, 8, remove_nans = True, keep_length = True)
    ratio4b = npl.running_mean(ratio3b, 8, remove_nans = True, keep_length = True)

    #ax.plot(ratio1, all_alts, label = 'ratio fomi', color = cols[0])
    ax.plot(ratio1abs, all_alts, label = 'ratio fomi abs', color = cols[0], linestyle = '--')
    ax.plot(ratio1run, all_alts, label = 'ratio fomi run', color = cols[0])

    ax.plot(ratio3a, all_alts, label = 'ratio new abs a', color = cols[2], linestyle = '--')
    ax.plot(ratio3b, all_alts, label = 'ratio new abs b', color = cols[4], linestyle = '--')

    ax.plot(ratio4a, all_alts, label = 'ratio new run a', color = cols[2])
    ax.plot(ratio4b, all_alts, label = 'ratio new run b', color = cols[4])

    ax.set_title(atm)
    #ax.set_ylim(20, 90)
    ax.set_xscale('log')
    ax.set_xlim(0.1, 1000)
    ax.grid()

plt.subplots_adjust(bottom = 0.1)
fig.legend(loc = 'lower center')
fig.savefig(cart_out_2 + 'check_ratios_NLTE_lotr.pdf')

for ax in axes:
    #ax.set_xscale('linear')
    ax.set_ylim(40, 90)
fig.savefig(cart_out_2 + 'check_ratios_NLTE_lotr_zoom.pdf')


pres = atm_pt[('mle', 'pres')]
x = np.log(1000./pres)
n_alts_trlo = np.sum(x < 12.5)
n_alts_lte = np.sum(x < 10)
