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

# sys.path.insert(0, '/home/fedefab/Scrivania/Research/Dotto/Git/spect_robot/')
# sys.path.insert(0, '/home/fedefab/Scrivania/Research/Dotto/Git/pythall/')
sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
import spect_base_module as sbm
import spect_classes as spcl

kbc = const.k/(const.h*100*const.c) # 0.69503
kboltz = 1.38064853e-23 # J/K
E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition

cp = 1.005e7 # specific enthalpy dry air - erg g-1 K-1
#############################################################

# cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'
cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'

allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweigths = dict(zip(allatms, atmweigths))
allco2 = np.arange(1,7)

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v1.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt.p'))
n_alts = 40

from scipy.optimize import Bounds, minimize, least_squares
import newparam_lib as npl
#############################################################

############## MINIMUM WITH HEIGHT DEPENDENT FIT ###########################

# pickle.dump(tutti_cycle, open(cart_out+'tutti_cycle_vals_allco2_v2.p', 'w'))
# pickle.dump(best, open(cart_out+'best_uniform_allco2_v2.p', 'w'))
# pickle.dump(tutti_cycle_atx0, open(cart_out+'tutti_cycle_atx0_vals_allco2_High.p', 'w'))
# pickle.dump(best_atx0, open(cart_out+'best_atx0_allco2_High.p', 'w'))

############################################################################
cco2 = 1

tutti3 = pickle.load(open(cart_out+'tutti3_vals.p'))
tuttil3 = np.array([cu[1] for cu in tutti3])
ind = tuttil3.argmin()
print(tutti3[ind])

#result = least_squares(delta_xi_tot, tutti3[ind][0], jac=jacdelta_xi_tot, args=(cco2,), verbose=2, bounds = (0,20))
# result = least_squares(delta_xi_tot, tutti3[ind][0], args=(cco2,), verbose=2, bounds = (0,20))
# print(result)
# print('######################################################')
# sys.exit()

# result = least_squares(npl.delta_xi_tot, tutti3[ind][0], jac=npl.jacdelta_xi_all_x0s_fast, args=(cco2,), verbose=1, method = 'lm')
# print(result)
# print('######################################################')
# sys.exit()
ialt = 30
#result = least_squares(npl.delta_xi_at_x0, tutti3[ind][0], jac=npl.jacdelta_xi_at_x0, args=(cco2, ialt,), verbose=1, method = 'lm')
result = least_squares(npl.delta_xi_at_x0_tot, np.ones(6), args=(cco2, ialt,), verbose=1, method = 'trf', bounds = bounds, gtol = None)

sys.exit()

bounds = (np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), np.array([100, 100, 100, 100, 100, 100]))
result = least_squares(npl.delta_xi_at_x0, tutti3[ind][0], jac=npl.jacdelta_xi_at_x0, args=(cco2, ialt,), verbose=1, method = 'trf', bounds = bounds, gtol = None)
print(result)
print('######################################################')

result = least_squares(npl.delta_xi_at_x0, np.ones(6), jac=npl.jacdelta_xi_at_x0, args=(cco2, ialt,), verbose=1, method = 'trf', bounds = bounds, gtol = None)
print(result)
print('######################################################')
sys.exit()


result = least_squares(delta_xi_tot, tutti3[ind][0], jac=jacdelta_xi_tot, args=(cco2,), verbose=1, method = 'lm')
print(result)
print('######################################################')

sys.exit()
# x = np.ones(6)
# result = least_squares(stup_delta, xis_0, args=(x,), verbose=1, method = 'lm')
# print(result)
# print('######################################################')
#
print(delta_xi_tot(xis_0, cco2))
print(jacdelta_xi_tot(xis_0, cco2))
print('tioitoitioioio')
# result = least_squares(delta_xi_tot, xis_0, jac=jacdelta_xi_tot, bounds=(0, 1), args=(cco2,), verbose=1, method = 'dogbox')
# print(result)
# print('######################################################')

result = least_squares(delta_xi_at_x0, xis_0, jac = jacdelta_xi_at_x0, args=(cco2,10), verbose=1, method = 'lm')
print(result)
print('######################################################')

result = least_squares(delta_xi_tot, xis_0, args=(cco2,), verbose=1, method = 'lm')
print(result)
print('######################################################')
#
sys.exit()


cco2 = 1
ialt = 10
xis_0 = (0.36, 0.05, 0.05, 0.5, 0.02) # uno in meno
print(delta_xi_at_x0(xis_0, cco2, ialt))
print(jacdelta_xi_at_x0(xis_0, cco2, ialt))
print('tioitoitioioio')
result = least_squares(delta_xi_at_x0, xis_0, jac=jacdelta_xi_at_x0, bounds=(0, 1), args=(cco2, ialt), verbose=1, gtol = 1.e-30, method = 'dogbox')

all_res_co2 = dict()
for cco2 in allco2:
    xis_0 = (0.36, 0.05, 0.05, 0.5, 0.02) # uno in meno
    all_coeffs_co2 = dict()
    for atm in allatms:
        for nom in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
            all_coeffs_co2[(atm, nom)] = all_coeffs[(atm, cco2, nom)]

    all_hr_ref = [all_coeffs[(atm, cco2, 'hr_ref')][:n_alts] for atm in allatms]

    #result = optimize.curve_fit(hr_from_xi, all_coeffs_co2, all_hr_ref, p0 = tuple(xis_0))
    #result = minimize(delta_xi, xis_0, method = 'nelder-mead', options={'xtol': 1e-8, 'disp': True}, args=(cco2,))#, bounds = bounds)

    all_res = []
    for ialt in range(n_alts):
        result = least_squares(delta_xi_at_x0, xis_0, jac=jacdelta_xi_at_x0, bounds=(0, 1), args=(cco2, ialt), verbose=1)
        print(result)
        all_res.append(result)

    all_res_co2[cco2] = all_res
