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
cart_out_2 = cart_base + 'lavori/CO2_cooling/new_param/NLTE/'

import spect_base_module as sbm
import spect_classes as spcl
from scipy.optimize import Bounds, minimize, least_squares
import newparam_lib as npl

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

for atm in allatms:
    for cco2 in allco2:
        hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_nlte')]
        hr_ref[:n_alts] = all_coeffs[(atm, cco2, 'hr_ref')][:n_alts]
        all_coeffs_nlte[(atm, cco2, 'hr_ref')] = hr_ref

pickle.dump(all_coeffs_nlte, open(cart_out_2 + 'all_coeffs_NLTE.p', 'wb'))

#############################################################

#bounds = (0.1*np.ones(6), 100.*np.ones(6))
minb = 0.01*np.ones(6)
maxb = 100*np.ones(6)
#maxb[allatms.index('mls')] = 0.011
bounds = (minb, maxb)

allres_varfit = dict()
varfit_xis = dict()
varfit_xis_2 = dict()

thresloop = 1.e-8
nloops = 1000

xtol = 1.e-12
gtol = 1.e-12

# xis_a_start = np.ones(6)
# xis_b_start = np.ones(6)

xis_a_start = np.array([0.3, 0.1, 0.1, 0.4, 0.05, 0.05])*10
xis_b_start = np.array([0.3, 0.1, 0.1, 0.4, 0.05, 0.05])*10
#xis_a_start[allatms.index('mls')] = 0.01
#xis_b_start[allatms.index('mls')] = 0.01

#xis_a_start = np.array([9.5, 0.5, 0.5, 5.5, 1.6, 1.6]) # comes from old fit
#xis_b_start = np.array([9.5, 0.5, 0.5, 5.5, 1.6, 1.6])

##### Rescaling both a and b by hr_nlte/hr_lte
for cco2 in range(1,npl.n_co2prof+1):
    for ialt in range(npl.max_alts_curtis):
        doloop = True
        jloop = 1
        xis_b = None

        # Fomichev's atm weights
        while doloop and jloop < nloops: # loop on a and b fit
            if jloop == 1:
                xis_start = xis_a_start
            else:
                xis_start = xis_a
            cnam = 'afit'
            result = least_squares(npl.delta_xi_at_x0_afit, xis_start, jac=npl.jacdelta_xi_at_x0_afit, args=(cco2, ialt, xis_b, atmweigths, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
            print(cco2, ialt, cnam, jloop, result.x)
            xis_a = result.x

            if jloop > 1:
                xis_old = varfit_xis[(cco2, ialt, cnam)]
                if np.mean(np.abs(xis_a - xis_old)) < thresloop:
                    doloop = False

            varfit_xis[(cco2, ialt, cnam)] = xis_a
            # xis_a_alts = [varfit_xis[(cco2, ialt, 'afit')] for ialt in nalt]
            # xis_b_alts = [varfit_xis[(cco2, ialt, 'bfit')] for ialt in nalt]
            # agn = npl.coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'acoeff', all_coeffs = all_coeffs_nlte)
            # agn_surf = npl.coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'asurf', all_coeffs = all_coeffs_nlte)
            # all_coeffs_nlte[(atm, cco2, 'acoeff')][..., ialt] = agn
            # all_coeffs_nlte[(atm, cco2, 'asurf')][ialt] = agn_surf

            if jloop == 1:
                xis_start = xis_b_start
            else:
                xis_start = xis_b
            cnam = 'bfit'
            result = least_squares(npl.delta_xi_at_x0_bfit, xis_start, jac=npl.jacdelta_xi_at_x0_bfit, args=(cco2, ialt, xis_a, atmweigths, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
            print(cco2, ialt, cnam, jloop, result.x)
            xis_b = result.x

            if jloop > 1:
                xis_old = varfit_xis[(cco2, ialt, cnam)]
                if np.mean(np.abs(xis_b - xis_old)) < thresloop:
                    doloop = False

            varfit_xis[(cco2, ialt, cnam)] = xis_b
            # bgn = npl.coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bcoeff', all_coeffs = all_coeffs_nlte)
            # bgn_surf = npl.coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bsurf', all_coeffs = all_coeffs_nlte)
            # all_coeffs_nlte[(atm, cco2, 'bcoeff')][..., ialt] = bgn
            # all_coeffs_nlte[(atm, cco2, 'bsurf')][ialt] = bgn_surf

            jloop += 1

print('######################################################')
pickle.dump(varfit_xis, open(cart_out_2+'varfit_NLTE_v4c.p', 'wb'))
# pickle.dump(all_coeffs_nlte, open(cart_out_2 + 'all_coeffs_NLTE_fitv4.p', 'wb'))

xis_a_start = np.ones(6)/6.
xis_b_start = np.ones(6)/6.
minb = 0.01*np.ones(6)
maxb = 100*np.ones(6)
# maxb[allatms.index('mls')] = 0.011
bounds = (minb, maxb)

all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

for cco2 in range(1,npl.n_co2prof+1):
    for ialt in range(npl.max_alts_curtis):
        doloop = True
        jloop = 1
        xis_b = None
        # Equal atm weights
        while doloop and jloop < nloops: # loop on a and b fit
            if jloop == 1:
                xis_start = xis_a_start
            else:
                xis_start = xis_a

            cnam = 'afit'
            result = least_squares(npl.delta_xi_at_x0_afit, xis_start, jac=npl.jacdelta_xi_at_x0_afit, args=(cco2, ialt, xis_b, atmweigths2, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
            print(cco2, ialt, cnam, jloop, result.x)
            xis_a = result.x

            if jloop > 1:
                xis_old = varfit_xis_2[(cco2, ialt, cnam)]
                if np.mean(np.abs(xis_a - xis_old)) < thresloop:
                    doloop = False

            varfit_xis_2[(cco2, ialt, cnam)] = xis_a
            # agn = npl.coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'acoeff', all_coeffs = all_coeffs_nlte)
            # agn_surf = npl.coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'asurf', all_coeffs = all_coeffs_nlte)
            # all_coeffs_nlte[(atm, cco2, 'acoeff')][..., ialt] = agn
            # all_coeffs_nlte[(atm, cco2, 'asurf')][ialt] = agn_surf

            if jloop == 1:
                xis_start = xis_b_start
            else:
                xis_start = xis_b
            cnam = 'bfit'
            result = least_squares(npl.delta_xi_at_x0_bfit, xis_start, jac=npl.jacdelta_xi_at_x0_bfit, args=(cco2, ialt, xis_a, atmweigths2, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
            print(cco2, ialt, cnam, jloop, result.x)
            xis_b = result.x

            if jloop > 1:
                xis_old = varfit_xis_2[(cco2, ialt, cnam)]
                if np.mean(np.abs(xis_b - xis_old)) < thresloop:
                    doloop = False

            varfit_xis_2[(cco2, ialt, cnam)] = xis_b
            # bgn = npl.coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bcoeff', all_coeffs = all_coeffs_nlte)
            # bgn_surf = npl.coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bsurf', all_coeffs = all_coeffs_nlte)
            # all_coeffs_nlte[(atm, cco2, 'bcoeff')][..., ialt] = bgn
            # all_coeffs_nlte[(atm, cco2, 'bsurf')][ialt] = bgn_surf

            jloop += 1

print('######################################################')
pickle.dump(varfit_xis_2, open(cart_out_2+'varfit_NLTE_v5c.p', 'wb'))

# pickle.dump(all_coeffs_nlte, open(cart_out_2 + 'all_coeffs_NLTE_fitv5.p', 'wb'))

# atmweigths3 = [0.25, 0.25, 0.25, 0.25, 0., 0.]
# atmweigths3 = dict(zip(allatms, atmweigths3))
# varfit_xis_3 = dict()
#
# all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))
#
# for cco2 in range(1,npl.n_co2prof+1):
#     for ialt in range(npl.n_alts_all):
#         doloop = True
#         jloop = 1
#         xis_b = None
#         # Equal atm weights
#         while doloop and jloop < nloops: # loop on a and b fit
#             if jloop == 1:
#                 xis_start = xis_a_start
#             else:
#                 xis_start = xis_a
#
#             cnam = 'afit'
#             result = least_squares(npl.delta_xi_at_x0_afit, xis_start, jac=npl.jacdelta_xi_at_x0_afit, args=(cco2, ialt, xis_b, atmweigths3, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
#             print(cco2, ialt, cnam, jloop, result.x)
#             xis_a = result.x
#
#             if jloop > 1:
#                 xis_old = varfit_xis_3[(cco2, ialt, cnam)]
#                 if np.mean(np.abs(xis_a - xis_old)) < thresloop:
#                     doloop = False
#
#             varfit_xis_3[(cco2, ialt, cnam)] = xis_a
#             # agn = npl.coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'acoeff', all_coeffs = all_coeffs_nlte)
#             # agn_surf = npl.coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'asurf', all_coeffs = all_coeffs_nlte)
#             # all_coeffs_nlte[(atm, cco2, 'acoeff')][..., ialt] = agn
#             # all_coeffs_nlte[(atm, cco2, 'asurf')][ialt] = agn_surf
#
#             if jloop == 1:
#                 xis_start = xis_b_start
#             else:
#                 xis_start = xis_b
#             cnam = 'bfit'
#             result = least_squares(npl.delta_xi_at_x0_bfit, xis_start, jac=npl.jacdelta_xi_at_x0_bfit, args=(cco2, ialt, xis_a, atmweigths3, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
#             print(cco2, ialt, cnam, jloop, result.x)
#             xis_b = result.x
#
#             if jloop > 1:
#                 xis_old = varfit_xis_3[(cco2, ialt, cnam)]
#                 if np.mean(np.abs(xis_b - xis_old)) < thresloop:
#                     doloop = False
#
#             varfit_xis_3[(cco2, ialt, cnam)] = xis_b
#             # bgn = npl.coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bcoeff', all_coeffs = all_coeffs_nlte)
#             # bgn_surf = npl.coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bsurf', all_coeffs = all_coeffs_nlte)
#             # all_coeffs_nlte[(atm, cco2, 'bcoeff')][..., ialt] = bgn
#             # all_coeffs_nlte[(atm, cco2, 'bsurf')][ialt] = bgn_surf
#
#             jloop += 1
#
# print('######################################################')
# pickle.dump(varfit_xis_3, open(cart_out_2+'varfit_NLTE_v6_noarctic.p', 'wb'))
#
# atmweigths4 = [0., 0., 0., 0., 0.5, 0.5]
# atmweigths4 = dict(zip(allatms, atmweigths4))
# varfit_xis_4 = dict()
#
# for cco2 in range(1,npl.n_co2prof+1):
#     for ialt in range(npl.n_alts_all):
#         doloop = True
#         jloop = 1
#         xis_b = None
#         # Equal atm weights
#         while doloop and jloop < nloops: # loop on a and b fit
#             if jloop == 1:
#                 xis_start = xis_a_start
#             else:
#                 xis_start = xis_a
#
#             cnam = 'afit'
#             result = least_squares(npl.delta_xi_at_x0_afit, xis_start, jac=npl.jacdelta_xi_at_x0_afit, args=(cco2, ialt, xis_b, atmweigths4, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
#             print(cco2, ialt, cnam, jloop, result.x)
#             xis_a = result.x
#
#             if jloop > 1:
#                 xis_old = varfit_xis_4[(cco2, ialt, cnam)]
#                 if np.mean(np.abs(xis_a - xis_old)) < thresloop:
#                     doloop = False
#
#             varfit_xis_4[(cco2, ialt, cnam)] = xis_a
#             # agn = npl.coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'acoeff', all_coeffs = all_coeffs_nlte)
#             # agn_surf = npl.coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'asurf', all_coeffs = all_coeffs_nlte)
#             # all_coeffs_nlte[(atm, cco2, 'acoeff')][..., ialt] = agn
#             # all_coeffs_nlte[(atm, cco2, 'asurf')][ialt] = agn_surf
#
#             if jloop == 1:
#                 xis_start = xis_b_start
#             else:
#                 xis_start = xis_b
#             cnam = 'bfit'
#             result = least_squares(npl.delta_xi_at_x0_bfit, xis_start, jac=npl.jacdelta_xi_at_x0_bfit, args=(cco2, ialt, xis_a, atmweigths4, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
#             print(cco2, ialt, cnam, jloop, result.x)
#             xis_b = result.x
#
#             if jloop > 1:
#                 xis_old = varfit_xis_4[(cco2, ialt, cnam)]
#                 if np.mean(np.abs(xis_b - xis_old)) < thresloop:
#                     doloop = False
#
#             varfit_xis_4[(cco2, ialt, cnam)] = xis_b
#             # bgn = npl.coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bcoeff', all_coeffs = all_coeffs_nlte)
#             # bgn_surf = npl.coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bsurf', all_coeffs = all_coeffs_nlte)
#             # all_coeffs_nlte[(atm, cco2, 'bcoeff')][..., ialt] = bgn
#             # all_coeffs_nlte[(atm, cco2, 'bsurf')][ialt] = bgn_surf
#
#             jloop += 1
#
# print('######################################################')
# pickle.dump(varfit_xis_4, open(cart_out_2+'varfit_NLTE_v7_arctic.p', 'wb'))
#
# bounds = (0.1*np.ones(6), np.ones(6))
# varfit_xis_4 = dict()
#
# for cco2 in range(1,npl.n_co2prof+1):
#     for ialt in range(npl.n_alts_all):
#         doloop = True
#         jloop = 1
#         xis_b = None
#         # Equal atm weights
#         while doloop and jloop < nloops: # loop on a and b fit
#             if jloop == 1:
#                 xis_start = xis_a_start
#             else:
#                 xis_start = xis_a
#
#             cnam = 'afit'
#             result = least_squares(npl.delta_xi_at_x0_afit, xis_start, jac=npl.jacdelta_xi_at_x0_afit, args=(cco2, ialt, xis_b, atmweigths, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
#             print(cco2, ialt, cnam, jloop, result.x)
#             xis_a = result.x
#
#             if jloop > 1:
#                 xis_old = varfit_xis_4[(cco2, ialt, cnam)]
#                 if np.mean(np.abs(xis_a - xis_old)) < thresloop:
#                     doloop = False
#
#             varfit_xis_4[(cco2, ialt, cnam)] = xis_a
#             # agn = npl.coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'acoeff', all_coeffs = all_coeffs_nlte)
#             # agn_surf = npl.coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'asurf', all_coeffs = all_coeffs_nlte)
#             # all_coeffs_nlte[(atm, cco2, 'acoeff')][..., ialt] = agn
#             # all_coeffs_nlte[(atm, cco2, 'asurf')][ialt] = agn_surf
#
#             if jloop == 1:
#                 xis_start = xis_b_start
#             else:
#                 xis_start = xis_b
#             cnam = 'bfit'
#             result = least_squares(npl.delta_xi_at_x0_bfit, xis_start, jac=npl.jacdelta_xi_at_x0_bfit, args=(cco2, ialt, xis_a, atmweigths, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'trf', bounds = bounds, gtol = gtol, xtol = xtol)
#             print(cco2, ialt, cnam, jloop, result.x)
#             xis_b = result.x
#
#             if jloop > 1:
#                 xis_old = varfit_xis_4[(cco2, ialt, cnam)]
#                 if np.mean(np.abs(xis_b - xis_old)) < thresloop:
#                     doloop = False
#
#             varfit_xis_4[(cco2, ialt, cnam)] = xis_b
#             # bgn = npl.coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bcoeff', all_coeffs = all_coeffs_nlte)
#             # bgn_surf = npl.coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bsurf', all_coeffs = all_coeffs_nlte)
#             # all_coeffs_nlte[(atm, cco2, 'bcoeff')][..., ialt] = bgn
#             # all_coeffs_nlte[(atm, cco2, 'bsurf')][ialt] = bgn_surf
#
#             jloop += 1
#
# print('######################################################')
# pickle.dump(varfit_xis_4, open(cart_out_2+'varfit_NLTE_v8_lim01.p', 'wb'))

# THE LM METHOD MAKES THE xis_b EXPLODE. why?
# for cco2 in range(1,npl.n_co2prof+1):
#     for ialt in range(npl.n_alts_all):
#         doloop = True
#         jloop = 1
#         xis_b = None
#         # Equal atm weights
#         while doloop and jloop < nloops: # loop on a and b fit
#             cnam = 'afit'
#             if jloop == 1:
#                 xis_start = xis_a_start
#             else:
#                 xis_start = xis_a
#             result = least_squares(npl.delta_xi_at_x0_afit, xis_start, jac=npl.jacdelta_xi_at_x0_afit, args=(cco2, ialt, xis_b, atmweigths2, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'lm')#, bounds = bounds, gtol = gtol, xtol = xtol)
#             print(cco2, ialt, cnam, jloop, result.x)
#             xis_a = result.x
#
#             if jloop > 1:
#                 xis_old = varfit_xis_2[(cco2, ialt, cnam)]
#                 if np.mean(np.abs(xis_a - xis_old)) < thresloop:
#                     doloop = False
#
#             varfit_xis_2[(cco2, ialt, cnam)] = xis_a
#
#             if jloop == 1:
#                 xis_start = xis_b_start
#             else:
#                 xis_start = xis_b
#             cnam = 'bfit'
#             result = least_squares(npl.delta_xi_at_x0_bfit, xis_start, jac=npl.jacdelta_xi_at_x0_bfit, args=(cco2, ialt, xis_a, atmweigths2, all_coeffs_nlte, 'hr_ref', ), verbose=1, method = 'lm')#, bounds = bounds, gtol = gtol, xtol = xtol)
#             print(cco2, ialt, cnam, jloop, result.x)
#             xis_b = result.x
#
#             if jloop > 1:
#                 xis_old = varfit_xis_2[(cco2, ialt, cnam)]
#                 if np.mean(np.abs(xis_b - xis_old)) < thresloop:
#                     doloop = False
#
#             varfit_xis_2[(cco2, ialt, cnam)] = xis_b
#
#             jloop += 1
#
# print('######################################################')
# pickle.dump(varfit_xis_2, open(cart_out_2+'varfit_NLTE_v6.p', 'wb'))

for iatmw in range(6):
    xis_a_start = np.ones(6) #zeros(6)+0.1
    xis_b_start = np.ones(6) #zeros(6)+0.1
    #xis_a_start[iatmw] = 10
    #xis_b_start[iatmw] = 10

    varfit_xis_2 = dict()
    for cco2 in range(1,npl.n_co2prof+1):
        for ialt in range(npl.n_alts_all):
            doloop = True
            jloop = 1
            xis_b = None

            # Weighting 1 only atmosphere iatmw
            weigatm = np.zeros(6)
            weigatm[iatmw] = 1.0
            weigatm = dict(zip(allatms, weigatm))

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
                print(cco2, ialt, cnam, jloop, result.x)
                xis_b = result.x

                if jloop > 1:
                    xis_old = varfit_xis_2[(cco2, ialt, cnam)]
                    if np.mean(np.abs(xis_b - xis_old)) < thresloop:
                        doloop = False

                varfit_xis_2[(cco2, ialt, cnam)] = xis_b

                jloop += 1

    print('######################################################')
    pickle.dump(varfit_xis_2, open(cart_out_2+'varfit_NLTE_iatm{}.p'.format(iatmw), 'wb'))
