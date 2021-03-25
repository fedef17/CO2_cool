#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
rcParams['figure.max_open_warning'] = 100
#import climtools_lib as ctl

from scipy import io
import scipy.constants as const
import pickle
from scipy.interpolate import PchipInterpolator as spline

if os.uname()[1] == 'ff-clevo':
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/SpectRobot/')
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/pythall/')
    cart_base = '/home/fedefab/Research/lavori/CO2_cooling/new_param/'
else:
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_base = '/home/fabiano/Research/lavori/CO2_cooling/new_param/'

cart_out = cart_base + 'LTE/'
cart_out_2 = cart_base + 'NLTE/'
cart_out_rep = cart_base + 'NLTE_reparam/'

import newparam_lib as npl
from eofs.standard import Eof
from sklearn.linear_model import LinearRegression

plt.rcParams['axes.axisbelow'] = True

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


all_alts = atm_pt[('mle', 'alts')]

pres = atm_pt[('mle', 'pres')]
x = np.log(1000./pres)
n_alts_trlo = np.sum(x < 12.5)
print(n_alts_trlo)
#print('low trans at {}'.format(alts[n_alts_trlo]))

all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

################################################################################

cco2 = 7

# plt.ion()
# for cco2 in [7]:#range(1,8):
#     ratios = [all_coeffs_nlte[(atm, cco2, 'hr_ref')]/all_coeffs_nlte[(atm, cco2, 'hr_lte')] for atm in allatms]
#     plt.figure()
#     for atm, col in zip(allatms, npl.color_set(6)):
#         plt.plot(all_coeffs_nlte[(atm, cco2, 'hr_ref')]-all_coeffs_nlte[(atm, cco2, 'hr_lte')], np.arange(len(ratios[0])), label = atm, color = col)
#         #plt.plot(all_coeffs_nlte[(atm, cco2, 'hr_lte')], np.arange(len(ratios[0])), label = atm, linestyle = '--', color = col)
#     plt.grid()
#     plt.legend()
#
# plt.figure()
# for atm, col in zip(allatms, npl.color_set(6)):
#     plt.plot(all_coeffs_nlte[(atm, cco2, 'hr_ref')]-all_coeffs_nlte[(atm, cco2, 'hr_lte')], np.arange(len(ratios[0])), label = atm, color = col)
#     #plt.plot(all_coeffs_nlte[(atm, cco2, 'hr_lte')], np.arange(len(ratios[0])), label = atm, linestyle = '--', color = col)
# plt.grid()
# plt.legend()

alt1 = 40
alt2 = 51

def prova(alt1, alt2):
    diffnlte = []
    hras = []
    hrbs = []
    temps = []
    tempsgrad = []
    for atm, col in zip(allatms, npl.color_set(6)):
        diffnlte.append(all_coeffs_nlte[(atm, cco2, 'hr_ref')][alt1:alt2]-all_coeffs_nlte[(atm, cco2, 'hr_lte')][alt1:alt2])
        hra, hrb = hra, hrb = npl.hr_from_ab_decomposed(all_coeffs[(atm, cco2, 'acoeff')], all_coeffs[(atm, cco2, 'bcoeff')], all_coeffs[(atm, cco2, 'asurf')], all_coeffs[(atm, cco2, 'bsurf')], atm_pt[(atm, 'temp')], atm_pt[(atm, 'surf_temp')], max_alts=66)
        hras.append(hra[alt1:alt2])
        hrbs.append(hrb[alt1:alt2])
        temps.append(atm_pt[(atm, 'temp')][alt1:alt2])
        tempsgrad.append(np.gradient(atm_pt[(atm, 'temp')])[alt1:alt2])

    hras = np.concatenate(hras)
    hrbs = np.concatenate(hrbs)
    temps = np.concatenate(temps)
    tempsgrad = np.concatenate(tempsgrad)
    diffnlte = np.concatenate(diffnlte)

    # Mod 1: uso solo hra e hrb
    X = np.stack([hras, hrbs]).T
    Y = np.stack(diffnlte)

    # STANDARDIZZO LE FEATURES
    #scaler = StandardScaler().fit(X)
    #X = scaler.transform(X)

    scores = []
    model1 = LinearRegression().fit(X, Y)
    print(model1.score(X, Y))
    scores.append(model1.score(X, Y))

    # Mod 2: butto dentro anche temperature
    X = np.stack([hras, hrbs, temps]).T
    model2 = LinearRegression().fit(X, Y)
    print(model2.score(X, Y))
    scores.append(model2.score(X, Y))

    # Mod 3: solo temp e tempgrad
    X = np.stack([hras, hrbs, temps, tempsgrad]).T
    model3 = LinearRegression().fit(X, Y)
    print(model3.score(X, Y))
    scores.append(model3.score(X, Y))

    # Mod 4: butto dentro temps
    X = np.stack([hras, temps]).T
    model4 = LinearRegression().fit(X, Y)
    print(model4.score(X, Y))
    scores.append(model4.score(X, Y))

    # Mod 5: solo temp e tempgrad
    X = np.stack([temps, tempsgrad]).T
    model5 = LinearRegression().fit(X, Y)
    print(model5.score(X, Y))
    scores.append(model5.score(X, Y))

    models = [model1, model2, model3, model4, model5]
    ##################################################################

    return scores

colors = npl.color_set(5)
allsco = []
for alt2 in range(41, 66):
    allsco.append(prova(alt1, alt2))

allsco = np.stack(allsco).T
plt.figure()
for i, (sco,col) in enumerate(zip(allsco, colors)):
    plt.plot(sco, np.arange(41,66), color = col, label = str(i))
plt.legend()
