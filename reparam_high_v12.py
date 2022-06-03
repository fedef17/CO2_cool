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

if os.uname()[1] == 'xaru':
    sys.path.insert(0, '/home/fedef/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fedef/Research/git/pythall/')
    cart_base = '/home/fedef/Research/lavori/CO2_cooling/new_param/'
else:
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_base = '/home/fabiano/Research/lavori/CO2_cooling/new_param/'

cart_out = cart_base + 'LTE/'
cart_out_2 = cart_base + 'NLTE/'
cart_out_rep = cart_base + 'NLTE_reparam/'
cart_out_3 = cart_base + 'NLTE_upper/'

cart_out_F = cart_base + 'newpar_allatm_v2/'
if not os.path.exists(cart_out_F): os.mkdir(cart_out_F)

import newparam_lib as npl
from eofs.standard import Eof
from sklearn.linear_model import LinearRegression
from scipy.optimize import Bounds, minimize, least_squares

import statsmodels.api as sm
from scipy import stats

plt.rcParams['axes.axisbelow'] = True

##############################################################
kbc = const.k/(const.h*100*const.c) # 0.69503
kboltz = 1.38064853e-23 # J/K
E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition

cp = 1.005e7 # specific enthalpy dry air - erg g-1 K-1
#############################################################

allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
#atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweights = np.ones(6)/6.
atmweights = dict(zip(allatms, atmweights))
allco2 = np.arange(1,npl.n_co2prof+1)

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v4.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt_v4.p'))


all_alts = atm_pt[('mle', 'alts')]

pres = atm_pt[('mle', 'pres')]
x = np.log(1000./pres)
n_alts_trlo = np.sum(x < 12.5)
print(n_alts_trlo)
#print('low trans at {}'.format(alts[n_alts_trlo]))

all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

cose_upper_atm = pickle.load(open(cart_out_3 + 'cose_upper_atm.p', 'rb'))
################################################################################

regrcoef = pickle.load(open(cart_out_rep + 'regrcoef_v3.p', 'rb'))
nlte_corr = pickle.load(open(cart_out_rep + 'nlte_corr_low.p', 'rb'))

alt2 = 51
#n_top = alt2 + 10
n_alts_cs = 80

#### NB!!!! questi sono diversi da quelli per la atm bassa!!!!
temps_anom = np.stack([atm_pt[(atm, 'temp')][alt2:]-np.mean(atm_pt[(atm, 'temp')][alt2:]) for atm in allatms])
atm_anom_mean = np.mean(temps_anom, axis = 0)
solver_anom = Eof(temps_anom)

x0 = solver_anom.pcs(pcscaling = 1)[:, 0] # questi sono uguali ai dotprods sotto
x02 = x0**2
x1 = solver_anom.pcs(pcscaling = 1)[:, 1]
x2 = solver_anom.pcs(pcscaling = 1)[:, 2]
x3 = solver_anom.pcs(pcscaling = 1)[:, 3]
x4 = solver_anom.pcs(pcscaling = 1)[:, 4]

do_single = False

alpha_dic = dict()
neste = 20
for n_top in [57, 60, 63, 65, 67, 70]:
    print('------------------- \n {} \n ---------------------'.format(n_top))
    ########### Qui c'è la parte del fit dell'alpha
    # alpha FIT!

    n_trans = n_top-alt2+1

    #bounds = (0.1*np.ones(n_trans), 100*np.ones(n_trans))
    bounds = (0.1*np.ones(n_trans), 100*np.ones(n_trans))

    for afit, atmw in zip(['a{}'.format(i) for i in range(5)], [np.ones(6), [0.3, 0.1, 0.1, 0.4, 0.05, 0.05], np.array([0.3, 0.1, 0.1, 0.4, 0.05, 0.05])**2, [0., 0.1, 0.1, 0., 1, 1], [1, 0., 0., 1, 0., 0.]]):
        atmweights = atmw

        alpha_unif = []
        alpha_unif_bf = []
        alpha_dic_atm = dict()
        start = np.ones(n_trans)
        name_escape_fun = 'L_esc_all_extP'
        for cco2 in range(1, npl.n_co2prof+1):
            result = least_squares(npl.delta_alpha_rec2_recf, start, args=(cco2, cose_upper_atm, alt2, n_top, atmweights, all_coeffs_nlte, atm_pt, name_escape_fun, ), verbose=1, method = 'trf', bounds = bounds, max_nfev = 20000, ftol = 1.e-10, gtol = 1.e-10, xtol = 1.e-10)
            alpha_unif.append(result.x)

            ### now the brute force version
            ### vamos entre 1 y 2 veces el valor encontrado arriba, con 10 steps
            ### cada altura indipendente? (va a costar mas)
            alpha_range = (np.ones(len(result.x)), 2*result.x)

            dic1 = dict()
            for este1 in range(neste):
                for este2 in range(neste):
                    #alp = np.ones(imaxcalc-alt2) + este*(alpha_range[1][:imaxcalc-alt2]-alpha_range[0][:imaxcalc-alt2])/neste

                    alp = np.append(este1*(alpha_range[1][0]-alpha_range[0][0])/neste, este2*(alpha_range[1][1]-alpha_range[0][1])/neste)

                    resu = npl.delta_alpha_rec2_recf(alp, cco2, cose_upper_atm, alt2, alt2+1, atmweights, all_coeffs_nlte, atm_pt, imaxcalc = alt2+2)
                    #np.sqrt(np.mean(resu**2))
                    cost = np.sum(resu**2)
                    dic1[(este1, este2)] = cost

            calpall = [dic1[ke] for ke in dic1]
            este1, este2 = list(dic1.keys())[np.argmin(calpall)]
            alp0in = np.append(este1*(alpha_range[1][0]-alpha_range[0][0])/neste, este2*(alpha_range[1][1]-alpha_range[0][1])/neste)
            alp0 = alp0in.copy()

            for imaxcalc in range(alt2+3, n_top+1):
                costall = []
                for este in range(neste):
                    #alp = np.ones(imaxcalc-alt2) + este*(alpha_range[1][:imaxcalc-alt2]-alpha_range[0][:imaxcalc-alt2])/neste

                    alp = np.append(alp0, este*(alpha_range[1][imaxcalc-alt2]-alpha_range[0][imaxcalc-alt2])/neste)

                    resu = npl.delta_alpha_rec2_recf(alp, cco2, cose_upper_atm, alt2, imaxcalc-1, atmweights, all_coeffs_nlte, atm_pt, imaxcalc = imaxcalc)
                    #np.sqrt(np.mean(resu**2))
                    cost = np.sum(resu**2)
                    costall.append(cost)

                estok = np.argmin(costall)
                print(imaxcalc, estok)
                alp = np.append(alp0, estok*(alpha_range[1][imaxcalc-alt2]-alpha_range[0][imaxcalc-alt2])/neste)

                alp0 = alp

            alpha_unif_bf.append(alp)

        alpha_unif = np.stack(alpha_unif)
        alpha_unif_bf = np.stack(alpha_unif_bf)
        pickle.dump(alpha_unif, open(cart_out_rep + 'alpha_unif_v{}_top{}.p'.format(afit[-1], n_top), 'wb'))
        alpha_dic[(afit, n_top)] = alpha_unif

        pickle.dump(alpha_unif, open(cart_out_rep + 'alpha_unif_v{}bf_top{}.p'.format(afit[-1], n_top), 'wb'))
        alpha_dic[(afit+'bf', n_top)] = alpha_unif_bf

    pickle.dump(alpha_dic, open(cart_out_rep + 'alpha_unif_allw_ntops.p', 'wb'))
