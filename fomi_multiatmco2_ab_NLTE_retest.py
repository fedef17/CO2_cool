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

if os.uname()[1] == 'ff-clevo':
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/SpectRobot/')
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/pythall/')
    cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'
elif os.uname()[1] == 'hobbes':
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'
    cart_out_2 = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE/'
else:
    raise ValueError('Unknown platform {}. Specify paths!'.format(os.uname()[1]))

import spect_base_module as sbm
import spect_classes as spcl

kbc = const.k/(const.h*100*const.c) # 0.69503
kboltz = 1.38064853e-23 # J/K
E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition

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

all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

from scipy.optimize import Bounds, minimize, least_squares
import newparam_lib as npl
#############################################################

tutti3 = pickle.load(open(cart_out+'tutti3_vals.p'))
tuttil3 = np.array([cu[1] for cu in tutti3])
ind = tuttil3.argmin()
#print(tutti3[ind])
start = tutti3[ind][0]

bounds = (np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), np.array([100, 100, 100, 100, 100, 100]))

allres_varfit = dict()
varfit_xis = dict()
varfit_xis_2 = dict()

thresloop = 1.e-8
nloops = 1000

xtol = 1.e-12
gtol = 1.e-12

xis_a_start = np.ones(6)
xis_b_start = np.ones(6)

##### Rescaling both a and b by hr_nlte/hr_lte
all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

# for cco2 in range(1,npl.n_co2prof+1):
#     for atm in allatms:
#         hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_nlte')]
#         hr_ref[:40] = all_coeffs_nlte[(atm, cco2, 'hr_lte')][:40]
#         all_coeffs_nlte[(atm, cco2, 'hr_ref')] = hr_ref
#
# pickle.dump(all_coeffs_nlte, open(cart_out_2 + 'all_coeffs_NLTE.p', 'wb'))

xis_a_start = np.ones(6)
xis_b_start = np.ones(6)
# xis_a_start[iatmw] = 10
# xis_b_start[iatmw] = 10
for xisinit in [np.ones(6), [10., 1., 1., 1., 1., 1.], [100., 0.1, 0.1, 0.1, 0.1, 0.1]]:
    print('\n\n\n BAUUUUUUUUUUUUUUUUUUUUUUUUUUU \n\n\n', xisinit)
    xis_a_start = xisinit
    xis_b_start = xisinit

    for iatmw in [0]:#range(6):
        varfit_xis_2 = dict()
        # Weighting 1 only atmosphere iatmw
        weigatm = np.zeros(6)
        weigatm[iatmw] = 1.0
        weigatm = dict(zip(allatms, weigatm))

        for cco2 in [7]:#range(1,npl.n_co2prof+1):
            for ialt in [47]:#range(npl.n_alts_all):
                print('\n######################### ', ialt, ' #########################\n')
                doloop = True
                jloop = 1
                xis_b = None


                while doloop and jloop < nloops: # loop on a and b fit
                    cnam = 'afit'
                    if jloop == 1:
                        xis_start = xis_a_start
                    else:
                        xis_start = xis_a

                    result = least_squares(npl.delta_xi_at_x0_afit, xis_start, jac=npl.jacdelta_xi_at_x0_afit, args=(cco2, ialt, xis_b, weigatm, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
                    print(cco2, ialt, cnam, jloop, result.x)
                    xis_a = result.x

                    if jloop > 1:
                        xis_old = varfit_xis_2[(cco2, ialt, cnam)]
                        if np.mean(np.abs(xis_a - xis_old)) < thresloop:
                            doloop = False

                    varfit_xis_2[(cco2, ialt, cnam)] = xis_a

                    if jloop == 1:
                        xis_start = xis_b_start
                    else:
                        xis_start = xis_b
                    cnam = 'bfit'
                    result = least_squares(npl.delta_xi_at_x0_bfit, xis_start, jac=npl.jacdelta_xi_at_x0_bfit, args=(cco2, ialt, xis_a, weigatm, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
                    #print(cco2, ialt, cnam, jloop, result.x)
                    xis_b = result.x

                    if jloop > 1:
                        xis_old = varfit_xis_2[(cco2, ialt, cnam)]
                        if np.mean(np.abs(xis_b - xis_old)) < thresloop:
                            doloop = False

                    varfit_xis_2[(cco2, ialt, cnam)] = xis_b

                    jloop += 1

            print(ialt, xis_a, xis_b)

        print('######################################################')
        #pickle.dump(varfit_xis_2, open(cart_out_2+'varfit_NLTE_iatm{}.p'.format(iatmw), 'wb'))
