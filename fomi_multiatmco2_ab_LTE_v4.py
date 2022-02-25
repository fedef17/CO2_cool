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

if os.uname()[1] == 'xaru':
    cart_base = '/home/fedef/Research/'
elif os.uname()[1] == 'hobbes':
    cart_base = '/home/fabiano/Research/'
else:
    raise ValueError('Unknown platform {}. Specify paths!'.format(os.uname()[1]))

sys.path.insert(0, cart_base + 'git/SpectRobot/')
sys.path.insert(0, cart_base + 'git/pythall/')
cart_out = cart_base + 'lavori/CO2_cooling/new_param/LTE/'

import spect_base_module as sbm
import spect_classes as spcl

from scipy.optimize import Bounds, minimize, least_squares
import newparam_lib as npl

kbc = npl.kbc #const.k/(const.h*100*const.c) # 0.69503
kboltz = npl.kboltz #1.38064853e-23 # J/K
E_fun = npl.E_fun #667.3799 # cm-1 energy of the 0110 -> 0000 transition

cp = 1.005e7 # specific enthalpy dry air - erg g-1 K-1
#############################################################

allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweigths = dict(zip(allatms, atmweigths))

atmweigths2 = np.ones(6)/6.
atmweigths2 = dict(zip(allatms, atmweigths2))

allco2 = np.arange(1,npl.n_co2prof+1)

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v4.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt_v4.p'))
n_alts = 40

#############################################################

# tutti3 = pickle.load(open(cart_out+'tutti3_vals.p'))
# tuttil3 = np.array([cu[1] for cu in tutti3])
# ind = tuttil3.argmin()
# print(tutti3[ind])
#
# start = tutti3[ind][0]
start = np.ones(6)

bounds = (np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), np.array([100, 100, 100, 100, 100, 100]))

allres_varfit = dict()
varfit_xis = dict()
varfit_xis_2 = dict()

thresloop = 0.001
nloops = 10

for cco2 in range(1,npl.n_co2prof+1):
    for ialt in range(npl.n_alts_all):
        doloop = True
        jloop = 1
        xis_b = None

        # Fomichev's atm weights
        while doloop and jloop < nloops: # loop on a and b fit
            cnam = 'afit'
            result = least_squares(npl.delta_xi_at_x0_afit, start, jac=npl.jacdelta_xi_at_x0_afit, args=(cco2, ialt, xis_b, atmweigths, ), verbose=1, method = 'trf', bounds = bounds)#, gtol = None, xtol = None)
            print(cco2, ialt, cnam, jloop, result.x)
            xis_a = result.x

            if jloop > 1:
                xis_old = varfit_xis[(cco2, ialt, cnam)]
                if np.mean(np.abs(xis_a - xis_old)) < thresloop:
                    doloop = False

            varfit_xis[(cco2, ialt, cnam)] = xis_a

            cnam = 'bfit'
            result = least_squares(npl.delta_xi_at_x0_bfit, start, jac=npl.jacdelta_xi_at_x0_bfit, args=(cco2, ialt, xis_a, atmweigths, ), verbose=1, method = 'trf', bounds = bounds)#, gtol = None, xtol = None)
            print(cco2, ialt, cnam, jloop, result.x)
            xis_b = result.x

            if jloop > 1:
                xis_old = varfit_xis[(cco2, ialt, cnam)]
                if np.mean(np.abs(xis_b - xis_old)) < thresloop:
                    doloop = False

            varfit_xis[(cco2, ialt, cnam)] = xis_b
            jloop += 1

print('######################################################')
pickle.dump(varfit_xis, open(cart_out+'varfit_LTE_v4.p', 'wb'))

for cco2 in range(1,npl.n_co2prof+1):
    for ialt in range(npl.n_alts_all):
        doloop = True
        jloop = 1
        xis_b = None
        # Equal atm weights
        while doloop and jloop < nloops: # loop on a and b fit
            cnam = 'afit'
            result = least_squares(npl.delta_xi_at_x0_afit, np.ones(6), jac=npl.jacdelta_xi_at_x0_afit, args=(cco2, ialt, xis_b, atmweigths2, ), verbose=1, method = 'trf', bounds = bounds)#, gtol = None, xtol = None)
            print(cco2, ialt, cnam, jloop, result.x)
            xis_a = result.x

            if jloop > 1:
                xis_old = varfit_xis_2[(cco2, ialt, cnam)]
                if np.mean(np.abs(xis_a - xis_old)) < thresloop:
                    doloop = False

            varfit_xis_2[(cco2, ialt, cnam)] = xis_a

            cnam = 'bfit'
            result = least_squares(npl.delta_xi_at_x0_bfit, np.ones(6), jac=npl.jacdelta_xi_at_x0_bfit, args=(cco2, ialt, xis_a, atmweigths2, ), verbose=1, method = 'trf', bounds = bounds)#, gtol = None, xtol = None)
            print(cco2, ialt, cnam, jloop, result.x)
            xis_b = result.x

            if jloop > 1:
                xis_old = varfit_xis_2[(cco2, ialt, cnam)]
                if np.mean(np.abs(xis_b - xis_old)) < thresloop:
                    doloop = False

            varfit_xis_2[(cco2, ialt, cnam)] = xis_b
            jloop += 1

print('######################################################')
pickle.dump(varfit_xis_2, open(cart_out+'varfit_LTE_v5.p', 'wb'))
