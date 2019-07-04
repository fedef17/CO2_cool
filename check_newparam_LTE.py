#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
#import climtools_lib as ctl

from scipy import io
import scipy.constants as const
import pickle

sys.path.insert(0, '/home/fedefab/Scrivania/Research/Dotto/Git/spect_robot/')
sys.path.insert(0, '/home/fedefab/Scrivania/Research/Dotto/Git/pythall/')
import spect_base_module as sbm
import spect_classes as spcl

import newparam_lib as npl

##############################################################
kbc = const.k/(const.h*100*const.c) # 0.69503
kboltz = 1.38064853e-23 # J/K
E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition

cp = 1.005e7 # specific enthalpy dry air - erg g-1 K-1
#############################################################

cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'

allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweigths = dict(zip(allatms, atmweigths))
allco2 = np.arange(1,7)

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v1.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt.p'))

n_alts = 55

best = pickle.load(open(cart_out+'best_uniform_allco2.p'))
alts = atm_pt[('mle', 'alts')][:n_alts]

figs = []
a0s = []
a1s = []
for cco2 in allco2:
    xis = best[cco2]/np.sum(best[cco2])
    print(cco2, xis)
    for atm in allatms:
        hr_ref = all_coeffs[(atm, cco2, 'hr_ref')][:n_alts]
        hr_calc = npl.hr_from_xi(xis, atm, cco2, all_coeffs = all_coeffs, atm_pt = atm_pt, allatms = allatms, n_alts = n_alts)
        hr_ab_orig = npl.hr_atm_calc(atm, cco2)[:n_alts]

        tit = 'co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'Alt (km)'
        hrs = [hr_ref, hr_ab_orig, hr_calc]
        labels = ['ref', 'calc', 'fit']
        fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-2., 2.))
        # fig = plt.figure()
        # plt.plot(hr_ref, alts, label = 'ref')
        # plt.plot(hr_calc, alts, label = 'calc')
        # plt.plot(hr_calc-hr_ref, alts, label = 'diff')
        # plt.legend()
        # plt.grid()
        # plt.title('co2: {} - atm: {}'.format(cco2, atm))
        figs.append(fig)
        a0s.append(a0)
        a1s.append(a1)

npl.adjust_ax_scale(a0s)
npl.adjust_ax_scale(a1s)
npl.plot_pdfpages(cart_out + 'check_newparam_LTE_unifit.pdf', figs)
