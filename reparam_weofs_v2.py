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
    cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'
else:
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'
    cart_out_2 = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE/'
    cart_out_rep = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE_reparam/'

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
n_alts = 55
#for n_alts in [41, 46, 51, 56, 61, 66]:
print(n_alts)
alts = atm_pt[('mle', 'alts')]

cartou = cart_out_rep + 'alts{}/'.format(n_alts)
if not os.path.exists(cartou): os.mkdir(cartou)

temps = [atm_pt[(atm, 'temp')][:n_alts] for atm in allatms]
temps = np.stack(temps)

temps_anom = np.stack([atm_pt[(atm, 'temp')][:n_alts]-np.mean(atm_pt[(atm, 'temp')][:n_alts]) for atm in allatms])
atm_anom_mean = np.mean(temps_anom, axis = 0)

solver_anom = Eof(temps_anom)
surftemps = np.array([atm_pt[(atm, 'surf_temp')] for atm in allatms])
surfanom = surftemps-np.mean(surftemps)

# ok so, if keeping only first and second eof I'm able to explain quite a fraction of the variability
# the coeffs will be written as: C = C0 + alpha*C1 + beta*C2, with C1 and C2 being the pcs of the actual temp profile with respect to the first two eofs. Calculation of C1 and C2 implies two dot products over 66 altitudes. Plus the sum to determine C. Affordable? yes!

# Now for the coeffs. Are the coeffs pcs linked to the temp pcs? (correlation?). If so, the method could work well!

### LTE COEFFS

# SIMPLER. Linear (or nonlinear?) regression of coeff with first pc of temp profile
regrcoef = dict()
for cco2 in range(1,8):
    for conam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        acos = np.stack([all_coeffs[(atm, cco2, conam)] for atm in allatms]) ### LTE COEFFS!
        if acos.ndim == 3:
            x0 = solver_anom.pcs(pcscaling = 1)[:, 0] # questi sono uguali ai dotprods sotto
            x1 = solver_anom.pcs(pcscaling = 1)[:, 1] # questi sono uguali ai dotprods sotto
            #acos = acos[:, :n_alts, ...][..., :n_alts]

            Y = acos - np.mean(acos, axis = 0)
            Y = Y.reshape(6, acos.shape[1]*acos.shape[2])
            X = np.stack([x0, x1]).T
            model1 = LinearRegression().fit(X, Y)
            print(model1.score(X, Y))

            regrcoef[(cco2, conam, 'mean')] = np.mean(acos, axis = 0)
            regrcoef[(cco2, conam, 'm1')] = np.reshape(model1.coef_[:, 0], acos[0].shape)
            regrcoef[(cco2, conam, 'm2')] = np.reshape(model1.coef_[:, 1], acos[0].shape)

            corrco = np.empty_like(acos[0])
            for i in range(acos[0].shape[0]):
                for j in range(acos[0].shape[1]):
                    corrco[i,j] = np.corrcoef(x0, acos[:, i, j])[1,0]
        else:
            x0 = surfanom # per i cosi surface uso la anomaly di surface temperature
            #acos = acos[:, :n_alts]

            corrco = np.empty_like(acos[0])
            for i in range(acos[0].shape[0]):
                corrco[i] = np.corrcoef(x0, acos[:, i])[1,0]

        cico, regrco, _, _ = npl.linearregre_coeff(x0, acos)

        regrcoef[(cco2, conam, 'R')] = corrco
        regrcoef[(cco2, conam, 'c')] = cico
        regrcoef[(cco2, conam, 'm')] = regrco



# for conam in ['acoeff', 'bcoeff']:
#     fig = plt.figure()
#     for ialt, col in zip(range(n_alts), npl.color_set(n_alts)):
#         plt.plot(np.abs(regrcoef[(cco2, conam, 'R')][:, ialt]), alts, color = col)
#     #plt.xlim(-0.02, 0.02)
#     plt.title(conam + ' - rcorr')
#     fig.savefig(cartou + '{}_rcorr.pdf'.format(conam))


# the scalar products between the temp anomalies and the first eof of the temperature profile
dotprods = np.array([np.dot(te-atm_anom_mean, solver_anom.eofs(eofscaling=1)[0]) for te in temps_anom])
dotprods1 = np.array([np.dot(te-atm_anom_mean, solver_anom.eofs(eofscaling=1)[1]) for te in temps_anom])


# figs = []
# for conam in ['acoeff', 'bcoeff']:#, 'asurf', 'bsurf']:
#     fig, axes = plt.subplots(figsize = (20,12), nrows=1, ncols=2, sharey=True)
#     coeff = regrcoef[(conam, 'c')]
#     mco = regrcoef[(conam, 'm')]
#     for ialt, col in zip(range(n_alts), npl.color_set(n_alts)):
#         axes[0].plot(coeff[:n_alts, ialt], alts, color = col)
#         axes[1].plot(mco[:n_alts, ialt], alts, color = col, linestyle = ':')
#         if ialt > 1 and ialt < n_alts-1:
#             if np.abs(coeff[ialt, ialt])/np.abs(np.mean([coeff[ialt-1, ialt-1], coeff[ialt+1, ialt+1]])) > 1.5:
#                 print('Atm {}. Unstable ialt {}'.format(atm, ialt))
#                 axes[0].plot(np.mean([coeff[:n_alts, ialt-1][:-2], coeff[:n_alts, ialt+1][2:]], axis = 0), alts[1:-1], color = col, linestyle = '--')
#             if np.abs(mco[ialt, ialt])/np.abs(np.mean([mco[ialt-1, ialt-1], mco[ialt+1, ialt+1]])) > 1.5:
#                 print('Atm {}. Unstable ialt {}'.format(atm, ialt))
#                 axes[1].plot(np.mean([mco[:n_alts, ialt-1][:-2], mco[:n_alts, ialt+1][2:]], axis = 0), alts[1:-1], color = col, linestyle = '--')
#     axes[0].set_title('c')
#     axes[1].set_title('m')
#     plt.suptitle(conam)
#     figs.append(fig)
# npl.plot_pdfpages(cartou + 'regrcoeff.pdf', figs)


# ### OK! now I use the dotprods and the regrcoef to reconstruct the a and b coeffs, and compute the hr and check wrt the reference and the varfit5.
cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'
tot_coeff_co2 = pickle.load(open(cart_out + 'tot_coeffs_co2_v2_LTE.p', 'rb'))

figs = []
a0s = []
a1s = []
for cco2 in range(1,8):
    for atm, dp, dp1, sa in zip(allatms, dotprods, dotprods1, surfanom):
        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]

        coeffs = dict()
        for conam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
            if 'surf' in conam:
                coeffs[conam] = regrcoef[(cco2, conam, 'c')] + regrcoef[(cco2, conam, 'm')]*sa
            else:
                coeffs[conam] = regrcoef[(cco2, conam, 'c')] + regrcoef[(cco2, conam, 'm')]*dp
                coeffs[(conam, 'v2')] = regrcoef[(cco2, conam, 'mean')] + regrcoef[(cco2, conam, 'm1')]*dp + regrcoef[(cco2, conam, 'm2')]*dp1

        hr_new = npl.hr_from_ab(coeffs['acoeff'], coeffs['bcoeff'], coeffs['asurf'], coeffs['bsurf'], temp, surf_temp, max_alts = 66)
        hr_new_v2 = npl.hr_from_ab(coeffs[('acoeff', 'v2')], coeffs[('bcoeff', 'v2')], coeffs['asurf'], coeffs['bsurf'], temp, surf_temp, max_alts = 66)

        tip = 'varfit5'
        acoeff = tot_coeff_co2[(tip, 'acoeff', cco2)]
        bcoeff = tot_coeff_co2[(tip, 'bcoeff', cco2)]
        asurf = tot_coeff_co2[(tip, 'asurf', cco2)]
        bsurf = tot_coeff_co2[(tip, 'bsurf', cco2)]

        hr_vf5 = npl.hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = 66)

        hr_ref = all_coeffs[(atm, cco2, 'hr_ref')]

        labels = ['ref', 'veof', 'veof2', 'vf5']
        hrs = [hr_ref, hr_new, hr_new_v2, hr_vf5]
        tit = 'co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'Alt (km)'
        #labels = ['ref'] + alltips + ['fomi rescale (no fit)', 'old param']
        fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-3, 3), xlim = (-40, 10), ylim = (10, 90), linestyles = ['-', '--', '--', ':'])

        figs.append(fig)
        a0s.append(a0)
        a1s.append(a1)

    npl.adjust_ax_scale(a0s)
    npl.adjust_ax_scale(a1s)
    npl.plot_pdfpages(cart_out_rep + 'check_reparam_LTE.pdf', figs)
