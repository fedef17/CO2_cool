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

from scipy.optimize import Bounds, minimize, least_squares
import newparam_lib as npl
#############################################################

tutti3 = pickle.load(open(cart_out+'tutti3_vals.p'))
tuttil3 = np.array([cu[1] for cu in tutti3])
ind = tuttil3.argmin()
print(tutti3[ind])

start = tutti3[ind][0]

bounds = (np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), np.array([100, 100, 100, 100, 100, 100]))

allres_varfit = dict()
varfit_xis = dict()
varfit_xis_2 = dict()

for cco2 in range(1,npl.n_co2prof+1):
    for ialt in range(npl.n_alts_all):
        # result = least_squares(npl.delta_xi_at_x0, np.ones(6), jac=npl.jacdelta_xi_at_x0, args=(cco2, ialt,), verbose=1, method = 'trf', bounds = bounds, gtol = None)
        # print(ialt, result.x)
        # allres_varfit[(cco2, ialt)] = result
        # varfit_xis[(cco2, ialt)] = result.x

        result = least_squares(npl.delta_xi_at_x0, start, jac=npl.jacdelta_xi_at_x0, args=(cco2, ialt, atmweigths, ), verbose=1, method = 'trf', bounds = bounds)#, gtol = None, xtol = None)
        print(cco2, ialt, result.x)
        varfit_xis[(cco2, ialt)] = result.x

        result = least_squares(npl.delta_xi_at_x0, np.ones(6), jac=npl.jacdelta_xi_at_x0, args=(cco2, ialt, atmweigths2, ), verbose=1, method = 'trf', bounds = bounds)#, gtol = None, xtol = None)
        print(cco2, ialt, result.x)
        varfit_xis_2[(cco2, ialt)] = result.x

print('######################################################')
pickle.dump(varfit_xis, open(cart_out+'varfit_LTE_v2b.p', 'wb'))
pickle.dump(varfit_xis_2, open(cart_out+'varfit_LTE_v3b.p', 'wb'))
