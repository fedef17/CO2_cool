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

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v1.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt.p'))
n_alts = 40

from scipy.optimize import Bounds, minimize, least_squares

#############################################################

def hr_atm_calc(atm, cco2):
    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    acoeff = all_coeffs[(atm, cco2, 'acoeff')]
    bcoeff = all_coeffs[(atm, cco2, 'bcoeff')]
    asurf = all_coeffs[(atm, cco2, 'asurf')]
    bsurf = all_coeffs[(atm, cco2, 'bsurf')]

    hr = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)

    return hr

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


def hr_from_xi(xis, atm, cco2, all_coeffs = all_coeffs, atm_pt = atm_pt, allatms = allatms, n_alts = 40):
    """
    Calculates the HR from the acoeff and bcoeff of the different atmospheres, using the weights xis.
    """

    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    hr_somma = np.zeros(n_alts, dtype = float)
    for atmprim, xi in zip(allatms, xis):
        acoeff = all_coeffs[(atmprim, cco2, 'acoeff')]
        bcoeff = all_coeffs[(atmprim, cco2, 'bcoeff')]
        asurf = all_coeffs[(atmprim, cco2, 'asurf')]
        bsurf = all_coeffs[(atmprim, cco2, 'bsurf')]

        h_ab = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)
        #print(atm, xi, np.mean(h_ab))
        hr_somma += xi * h_ab[:n_alts]

    hr_somma = hr_somma/np.sum(xis)

    return hr_somma


def hr_from_xi_at_x0(xis, atm, cco2, ialt, all_coeffs = all_coeffs, atm_pt = atm_pt, allatms = allatms):
    """
    Calculates the HR from the acoeff and bcoeff of the different atmospheres, using the weights xis.
    """

    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    hr_somma = 0.
    for atmprim, xi in zip(allatms, xis):
        acoeff = all_coeffs[(atmprim, cco2, 'acoeff')]
        bcoeff = all_coeffs[(atmprim, cco2, 'bcoeff')]
        asurf = all_coeffs[(atmprim, cco2, 'asurf')]
        bsurf = all_coeffs[(atmprim, cco2, 'bsurf')]

        h_ab = hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, ialt)
        #print(atm, xi, np.mean(h_ab))
        hr_somma += xi * h_ab

    hr_somma = hr_somma/np.sum(xis)

    return hr_somma


def delta_xi(xis, cco2, n_alts = 40):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """

    fu = 0.
    for atm in allatms:
        hr = all_coeffs[(atm, cco2, 'hr_ref')][:n_alts]
        hr_somma = hr_from_xi(xis, atm, cco2)
        fu += atmweigths[atm] * np.sum((hr - hr_somma)**2)

    return fu


def delta_xi_tot_fomi(xis, cco2, n_alts = 40):
    """
    Modified delta function at page 511 bottom. Gives a vector with differences for each atm profile.
    """

    fu = 0.0
    for ialt in range(n_alts):
        fuialt = delta_xi_at_x0(xis, cco2, ialt, squared_residuals = True)
        fu += np.mean(fuialt)

    return fu

def delta_xi_tot(xis, cco2, n_alts = 40):
    """
    Modified delta function at page 511 bottom. Gives a vector with differences for each atm profile.
    """

    fu = np.zeros(len(allatms))
    for ialt in range(n_alts):
        fuialt = delta_xi_at_x0(xis, cco2, ialt)
        for i, atm in enumerate(allatms):
            fu[i] += fuialt[i]**2

    fu = np.sqrt(fu)

    return fu

def delta_xi_at_x0(xis, cco2, ialt, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths, squared_residuals = False):
    """
    This is done for a single altitude x0.
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """

    fu = np.zeros(len(allatms))
    for i, atm in enumerate(allatms):
        hr = all_coeffs[(atm, cco2, 'hr_ref')][ialt]
        #hr_somma = hr_from_xi(xis, atm, cco2)[ialt]
        hr_somma = hr_from_xi_at_x0(xis, atm, cco2, ialt)

        if not squared_residuals:
            fu[i] += atmweigths[atm] * (hr_somma - hr)
        else:
            fu[i] += atmweigths[atm] * (hr_somma - hr)**2

    return fu


def jacdelta_xi_tot(xis, cco2, n_alts = 40):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """

    J = np.empty((len(allatms), len(xis)))
    jacall = jacdelta_xi_all_x0s_fast(xis, cco2)
    delta = delta_xi_tot(xis, cco2)
    alldeltas = []
    for ialt in range(n_alts):
        alldeltas.append(delta_xi_at_x0(xis, cco2, ialt))

    for i in range(len(allatms)):
        for k in range(len(xis)):
            #print(i,k)
            J[i,k] = 1/(delta[i]) * np.sum([alldeltas[ialt][i]*jacall[i,k,ialt] for ialt in range(n_alts)])

    #print(np.mean(J))
    return J


def jacdelta_xi_at_x0(xis, cco2, ialt, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths):
    """
    Jacobian of delta_xi_at_x0.
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
            J[i,k] = atmweigths[allatms[i]]/np.sum(xis) * (hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, ialt) - hr_from_xi(xis, allatms[i], cco2)[ialt])
            #J[i,k] = atmweigths[allatms[i]] * hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, ialt)

    return J


def jacdelta_xi_all_x0s_fast(xis, cco2, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths):
    """
    Jacobian of delta_xi_at_x0.
    """

    J = np.empty((len(allatms), len(xis), n_alts))

    for i in range(len(allatms)):
        temp = atm_pt[(allatms[i], 'temp')]
        surf_temp = atm_pt[(allatms[i], 'surf_temp')]

        hrsomma = hr_from_xi(xis, allatms[i], cco2)
        for k in range(len(xis)):
            acoeff = all_coeffs[(allatms[k], cco2, 'acoeff')]
            bcoeff = all_coeffs[(allatms[k], cco2, 'bcoeff')]
            asurf = all_coeffs[(allatms[k], cco2, 'asurf')]
            bsurf = all_coeffs[(allatms[k], cco2, 'bsurf')]
            hrsing = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)
            for ialt in range(n_alts):
                J[i,k,ialt] = atmweigths[allatms[i]]/np.sum(xis) * (hrsing[ialt] - hrsomma[ialt])

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

#if __name__ == "__main__":

#####################################################################################
#####################################################################################
#### COMINCIA il MAIN

cco2 = 1

tutti_cycle = dict()
estval = 19
estval2 = 1
estval3 = 10
estval4 = 3

xis = np.array([estval, estval2, estval2, estval3, estval4, estval4])
coso = delta_xi_tot_fomi(xis, cco2)
coso2 = np.mean(delta_xi_tot(xis, cco2))
print(coso,coso2)
#
# best = dict()
# sto = 7
# #for n in [1., 0.1, 0.01]:
# for cco2 in range(1,7):
#     cosomin = 1.0
#     for n in [1.0, 0.1]:
#         tutti_n =  dict()
#         for val in np.arange(-sto*n,(sto+1)*n,n):
#             print(cco2, n, val)
#             for val2 in np.arange(-sto*n,(sto+1)*n,n):
#                 for val3 in np.arange(-sto*n,(sto+1)*n,n):
#                     for val4 in np.arange(-sto*n,(sto+1)*n,n):
#                         xis = np.array([estval+val, estval2+val2, estval2+val2, estval3+val3, estval4+val4, estval4+val4])
#                         if np.any(xis < 0.): continue
#                         #coso = np.mean(delta_xi_tot(xis, cco2))
#                         coso = delta_xi_tot_fomi(xis, cco2)
#                         #print(xis, coso)
#                         if coso < cosomin: #0.1126:
#                             tutti_n[(estval+val, estval2+val2, estval3+val3, estval4+val4)] = coso
#                             cosomin = coso
#
#         tutti_cycle[n] = tutti_n
#         tuttil = np.array(tutti_n.values())
#         ind = tuttil.argmin()
#         keok = tutti_n.keys()[ind]
#         print('best', keok, tutti_n[keok])
#         estval, estval2, estval3, estval4 = keok
#
#     best[cco2] = np.array(keok)[[0,1,1,2,3,3]]
#
# pickle.dump(tutti_cycle, open(cart_out+'tutti_cycle_vals_allco2.p', 'w'))
# pickle.dump(best, open(cart_out+'best_uniform_allco2.p', 'w'))
# sys.exit()

sto = 7
xis_best_uniform = np.array([0.5404814 , 0.00929978, 0.00929978, 0.26641138, 0.08725383, 0.08725383])
best_atx0 = dict()

tutti_cycle_atx0 = dict()
estval = 19.76
estval2 = 0.34
estval3 = 9.77
estval4 = 3.19

for cco2 in range(1,7):
    for ialt in range(n_alts):
        cosomin = 1.0
        for n in [1., 0.1]:
            tutti_n =  []
            for val in np.arange(-sto*n,(sto+1)*n,n):
                print(cco2, ialt, n, val)
                for val2 in np.arange(-sto*n,(sto+1)*n,n):
                    for val3 in np.arange(-sto*n,(sto+1)*n,n):
                        for val4 in np.arange(-sto*n,(sto+1)*n,n):
                            xis = np.array([estval+val, estval2+val2, estval2+val2, estval3+val3, estval4+val4, estval4+val4])
                            if np.any(xis < 0.): continue
                            coso = np.mean(delta_xi_at_x0(xis, cco2, ialt, squared_residuals = True))
                            #print(xis, coso)
                            if coso < cosomin:
                                tutti_n.append((xis, coso))
                                cosomin = coso

            tuttil = np.array([cu[1] for cu in tutti_n])
            #ind = tuttil.argmin()
            ordered = np.argsort(tuttil)
            ind = ordered[0]
            print('best', tutti_n[ind])
            tutti_cycle_atx0[(cco2, ialt, n)] = np.array(tutti_n)[ordered[:10]]
            keok = tutti_n[ind][0]

            estval, estval2, estval3, estval4 = keok[[0,1,3,4]]
            best_atx0[(cco2, ialt)] = np.array(keok)

for cco2 in range(1,7):
    su = 0.
    for ialt in range(n_alts):
        xis = best_atx0[(cco2, ialt)]
        su += np.mean(delta_xi_at_x0(xis, cco2, ialt, squared_residuals = True))

    print('final', su)

pickle.dump(tutti_cycle_atx0, open(cart_out+'tutti_cycle_atx0_vals_allco2.p', 'w'))
pickle.dump(best_atx0, open(cart_out+'best_atx0_allco2.p', 'w'))
sys.exit()
############################################################################

tutti3 = pickle.load(open(cart_out+'tutti3_vals.p'))
tuttil3 = np.array([cu[1] for cu in tutti3])
ind = tuttil3.argmin()
print(tutti3[ind])

#result = least_squares(delta_xi_tot, tutti3[ind][0], jac=jacdelta_xi_tot, args=(cco2,), verbose=2, bounds = (0,20))
result = least_squares(delta_xi_tot, tutti3[ind][0], args=(cco2,), verbose=2, bounds = (0,20))
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
