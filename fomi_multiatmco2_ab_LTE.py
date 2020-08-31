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

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v2.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt_v2.p'))
n_alts = 40

from scipy.optimize import Bounds, minimize, least_squares

#############################################################

def hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp):
    n_alts = len(temp)
    epsilon_ab_tot = np.zeros(n_alts, dtype = float)

    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    for xi in range(n_alts):
        epsilon_ab_tot[xi] = np.sum((acoeff[:, xi] + bcoeff[:, xi]* phi_fun[xi]) * phi_fun) # il contributo della colonna
        epsilon_ab_tot[xi] += (asurf[xi] + bsurf[xi]* phi_fun[xi]) * phi_fun_g

    return epsilon_ab_tot


def hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, x0):
    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    epsilon_ab_tot = np.sum((acoeff[:, x0] + bcoeff[:, x0]* phi_fun[x0]) * phi_fun) # il contributo della colonna
    epsilon_ab_tot += (asurf[x0] + bsurf[x0]* phi_fun[x0]) * phi_fun_g

    return epsilon_ab_tot


def hr_from_xi(all_coeffs_co2, xis_5, atm_pt = atm_pt, allatms = allatms, n_alts = 40):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """
    xis = list(xis_5) + [1.-np.sum(xis_5)]

    all_hr_somma = []
    for atm in allatms:
        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]

        hr_somma = np.zeros(n_alts, dtype = float)
        for atmprim, xi in zip(allatms, xis):
            acoeff = all_coeffs[(atmprim, 'acoeff')]
            bcoeff = all_coeffs[(atmprim, 'bcoeff')]
            asurf = all_coeffs[(atmprim, 'asurf')]
            bsurf = all_coeffs[(atmprim, 'bsurf')]

            h_ab = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)

            hr_somma += xi * h_ab[:n_alts]

        all_hr_somma.append(hr_somma)

    return all_hr_somma


def hr_from_xi6(xis, atm, atm_pt = atm_pt, allatms = allatms, n_alts = 40):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """

    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    hr_somma = np.zeros(n_alts, dtype = float)
    for atmprim, xi in zip(allatms, xis):
        acoeff = all_coeffs[(atmprim, 'acoeff')]
        bcoeff = all_coeffs[(atmprim, 'bcoeff')]
        asurf = all_coeffs[(atmprim, 'asurf')]
        bsurf = all_coeffs[(atmprim, 'bsurf')]

        h_ab = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)

        hr_somma += xi * h_ab[:n_alts]

    hr_somma = hr_somma/np.sum(xis)

    return hr_somma


def delta_xi(xis_5, cco2, n_alts = 40):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """
    xis = list(xis_5) + [1.-np.sum(xis_5)]

    print(cco2)
    print(xis)

    fu = 0.
    for atm in allatms:
        hr = all_coeffs[(atm, cco2, 'hr_ref')][:n_alts]

        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]

        hr_somma = np.zeros(n_alts, dtype = float)
        for atmprim, xi in zip(allatms, xis):
            acoeff = all_coeffs[(atmprim, cco2, 'acoeff')]
            bcoeff = all_coeffs[(atmprim, cco2, 'bcoeff')]
            asurf = all_coeffs[(atmprim, cco2, 'asurf')]
            bsurf = all_coeffs[(atmprim, cco2, 'bsurf')]

            h_ab = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)

            hr_somma += xi * h_ab[:n_alts]

        fu += atmweigths[atm] * np.sum((hr - hr_somma)**2)

    return fu

def delta_xi_tot(xis_5, cco2, n_alts = 40):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """
    xis = list(xis_5) + [1.-np.sum(xis_5)]

    fu = np.zeros(len(allatms))
    for ialt in range(n_alts):
        fuialt = delta_xi_at_x0(xis_5, cco2, ialt)
        for i, atm in enumerate(allatms):
            fu[i] += fuialt[i]**2

    fu = np.sqrt(fu)

    return fu

def delta_xi_at_x0(xis_5, cco2, ialt, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """
    xis = list(xis_5) + [1.-np.sum(xis_5)]

    print(cco2)
    print(xis)
    print(ialt)

    fu = np.zeros(len(allatms))
    for i, atm in enumerate(allatms):
        hr = all_coeffs[(atm, cco2, 'hr_ref')][ialt]

        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]

        hr_somma = 0.
        for atmprim, xi in zip(allatms, xis):
            acoeff = all_coeffs[(atmprim, cco2, 'acoeff')]
            bcoeff = all_coeffs[(atmprim, cco2, 'bcoeff')]
            asurf = all_coeffs[(atmprim, cco2, 'asurf')]
            bsurf = all_coeffs[(atmprim, cco2, 'bsurf')]

            h_ab = hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, ialt)

            hr_somma += xi * h_ab

        #fu += atmweigths[atm] * np.sum((hr - hr_somma)**2)
        fu[i] += atmweigths[atm] * (hr_somma - hr)

    return fu


def jacdelta_xi_tot(xis_5, cco2, n_alts = 40):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """
    xis = list(xis_5) + [1.-np.sum(xis_5)]

    J = np.empty((len(allatms), len(xis_5)))

    for i in range(len(allatms)):
        for k in range(len(xis_5)):
            J[i,k] = 1/(delta_xi_tot(xis_5, cco2)[i]) * np.sum([delta_xi_at_x0(xis_5, cco2, ialt)[i]*jacdelta_xi_at_x0(xis_5, cco2, ialt)[i,k] for ialt in range(n_alts)])

    return J


def jacdelta_xi_tot6(xis, cco2, n_alts = 40):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """

    J = np.empty((len(allatms), len(xis)))

    for i in range(len(allatms)):
        for k in range(len(xis)):
            J[i,k] = 1/(delta_xi_tot(xis, cco2)[i]) * np.sum([delta_xi_at_x0(xis, cco2, ialt)[i]*jacdelta_xi_at_x0_6(xis, cco2, ialt)[i,k] for ialt in range(n_alts)])

    return J


def jacdelta_xi_at_x0(xis_5, cco2, ialt, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """
    xis = list(xis_5) + [1.-np.sum(xis_5)]

    J = np.empty((len(allatms), len(xis_5)))

    acoeff6 = all_coeffs[(allatms[-1], cco2, 'acoeff')]
    bcoeff6 = all_coeffs[(allatms[-1], cco2, 'bcoeff')]
    asurf6 = all_coeffs[(allatms[-1], cco2, 'asurf')]
    bsurf6 = all_coeffs[(allatms[-1], cco2, 'bsurf')]

    for i in range(len(allatms)):
        temp = atm_pt[(allatms[i], 'temp')]
        surf_temp = atm_pt[(allatms[i], 'surf_temp')]

        for k in range(len(xis_5)):
            acoeff = all_coeffs[(allatms[k], cco2, 'acoeff')]
            bcoeff = all_coeffs[(allatms[k], cco2, 'bcoeff')]
            asurf = all_coeffs[(allatms[k], cco2, 'asurf')]
            bsurf = all_coeffs[(allatms[k], cco2, 'bsurf')]
            J[i,k] = atmweigths[allatms[i]] * (hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, ialt) - hr_from_ab_at_x0(acoeff6, bcoeff6, asurf6, bsurf6, temp, surf_temp, ialt))
            #J[i,k] = atmweigths[allatms[i]] * hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, ialt)

    return J


def jacdelta_xi_at_x0_6(xis, cco2, ialt, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """

    J = np.empty((len(allatms), len(xis)))

    for i in range(len(allatms)):
        temp = atm_pt[(allatms[i], 'temp')]
        surf_temp = atm_pt[(allatms[i], 'surf_temp')]

        for k in range(len(xis)):
            acoeff = all_coeffs[(allatms[k], cco2, 'acoeff')]
            bcoeff = all_coeffs[(allatms[k], cco2, 'bcoeff')]
            asurf = all_coeffs[(allatms[k], cco2, 'asurf')]
            bsurf = all_coeffs[(allatms[k], cco2, 'bsurf')]
            J[i,k] = atmweigths[allatms[i]] * (hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, ialt)/np.sum(xis) - hr_from_xi(acoeff6, bcoeff6, asurf6, bsurf6, temp, surf_temp, ialt))
            #J[i,k] = atmweigths[allatms[i]] * hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, ialt)

    return J


# >>> def model(x, u):
# ...     return x[0] * (u ** 2 + x[1] * u) / (u ** 2 + x[2] * u + x[3])

# >>> def fun(x, u, y):
# ...     return model(x, u) - y
#
# >>> def jac(x, u, y):
# ...     J = np.empty((u.size, x.size))
# ...     den = u ** 2 + x[2] * u + x[3]
# ...     num = u ** 2 + x[1] * u
# ...     J[:, 0] = num / den
# ...     J[:, 1] = x[0] * u / den
# ...     J[:, 2] = -x[0] * num * u / den ** 2
# ...     J[:, 3] = -x[0] * num / den ** 2
# ...     return J
#
# >>> u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1,
# ...               8.33e-2, 7.14e-2, 6.25e-2])
# >>> y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2,
# ...               4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2])
# >>> x0 = np.array([2.5, 3.9, 4.15, 3.9])
# >>> res = least_squares(fun, x0, jac=jac, bounds=(0, 100), args=(u, y), verbose=1)

######################################################################

# STEP 4 pagina 511: facciamo la media pesata tra le diverse atmosfere

###################################################################

# bounds = Bounds(([0, 1.0], [0, 1.0], [0, 1.0], [0, 1.0], [0, 1.0]))

# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
# res = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

cco2 = 1
xis_0 = (0.2, 0.2, 0.2, 0.2, 0.2) # uno in meno
print(delta_xi_tot(xis_0, cco2))
print(jacdelta_xi_tot(xis_0, cco2))
print('tioitoitioioio')
result = least_squares(delta_xi_tot, xis_0, jac=jacdelta_xi_tot, bounds=(0, 1), args=(cco2,), verbose=1, gtol = 1.e-20, method = 'dogbox')


for ii in np.lispace(0., 1., 10)



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
