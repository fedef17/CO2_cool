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

################################################################################

regrcoef = pickle.load(open(cart_out_rep + 'regrcoef_v3.p', 'rb'))

n_alts = 54
temps_anom = np.stack([atm_pt[(atm, 'temp')][:n_alts]-np.mean(atm_pt[(atm, 'temp')][:n_alts]) for atm in allatms])
atm_anom_mean = np.mean(temps_anom, axis = 0)
solver_anom = Eof(temps_anom)

x0 = solver_anom.pcs(pcscaling = 1)[:, 0] # questi sono uguali ai dotprods sotto
x1 = solver_anom.pcs(pcscaling = 1)[:, 1] # questi sono uguali ai dotprods sotto

surftemps = np.array([atm_pt[(atm, 'surf_temp')] for atm in allatms])
surfanom = surftemps-np.mean(surftemps)

alt1 = 40
alt2 = 51
cco2 = 7
def prova(cco2, alt1, alt2, use_model = 1):
    diffnlte = []
    hras = []
    hrbs = []
    temps = []
    tempsgrad = []
    for atm, col in zip(allatms, npl.color_set(6)):
        diffnlte.append(all_coeffs_nlte[(atm, cco2, 'hr_ref')][alt1:alt2]-all_coeffs_nlte[(atm, cco2, 'hr_lte')][alt1:alt2])
        hra, hrb = npl.hr_from_ab_diagnondiag(all_coeffs[(atm, cco2, 'acoeff')], all_coeffs[(atm, cco2, 'bcoeff')], all_coeffs[(atm, cco2, 'asurf')], all_coeffs[(atm, cco2, 'bsurf')], atm_pt[(atm, 'temp')], atm_pt[(atm, 'surf_temp')], max_alts=npl.n_alts_all)
        hras.append(hra[alt1:alt2])
        hrbs.append(hrb[alt1:alt2])
        temps.append(atm_pt[(atm, 'temp')][alt1:alt2])
        tempsgrad.append(np.gradient(atm_pt[(atm, 'temp')])[alt1:alt2])

    hras = np.stack(hras)
    hrbs = np.stack(hrbs)
    temps = np.stack(temps)
    tempsgrad = np.stack(tempsgrad)
    diffnlte = np.stack(diffnlte)

    #### devo splittare io nelle altezze.
    ints = []
    coefs = []
    for ii in range(hras.shape[1]):
        # Mod 1: uso solo hra e hrb
        X = np.stack([hras[:,ii], hrbs[:,ii], x0, x1]).T
        Y = np.stack(diffnlte[:,ii])

        scores = []
        model1 = LinearRegression().fit(X, Y)
        print(ii, model1.score(X, Y))
        scores.append(model1.score(X, Y))

        ints.append(model1.intercept_)
        coefs.append(model1.coef_)
        print(model1.intercept_, model1.coef_)

    return ints, coefs

#    return model1.intercept_, model1.coef_

# plt.ion()
# fig2, ax2 = plt.subplots()
# for i, (atm, col) in enumerate(zip(allatms, npl.color_set(6))):
#     diff = all_coeffs_nlte[(atm, cco2, 'hr_ref')][alt1:alt2]-all_coeffs_nlte[(atm, cco2, 'hr_lte')][alt1:alt2]
#     hra, hrb = npl.hr_from_ab_diagnondiag(all_coeffs[(atm, cco2, 'acoeff')], all_coeffs[(atm, cco2, 'bcoeff')], all_coeffs[(atm, cco2, 'asurf')], all_coeffs[(atm, cco2, 'bsurf')], atm_pt[(atm, 'temp')], atm_pt[(atm, 'surf_temp')], max_alts=npl.n_alts_all)
#     hr_nlte_corr = nlte_corr[(cco2, 'c')] + nlte_corr[(cco2, 'm1')] * hra[alt1:alt2] + nlte_corr[(cco2, 'm2')] * hrb[alt1:alt2] + nlte_corr[(cco2, 'm3')] * x0[i] + nlte_corr[(cco2, 'm4')] * x1[i]
#
#     ax2.plot(diff, np.arange(alt1, alt2), color = col, label = atm)
#     ax2.plot(hr_nlte_corr, np.arange(alt1, alt2), color = col, linestyle = '--')
#     ax2.plot(diff-hr_nlte_corr, np.arange(alt1, alt2), color = col, linestyle = ':')
#     #ax2.plot(hra[alt1:alt2], np.arange(alt1, alt2), color = col, linestyle = '--')
#     #ax2.plot(hrb[alt1:alt2], np.arange(alt1, alt2), color = col, linestyle = ':')
#
# ax2.grid()
# ax2.legend()
# fig2.savefig(cart_out_rep + 'Test_lowtrans_v3.pdf')

#sys.exit()

nlte_corr = dict()
for cco2 in range(1,npl.n_co2prof+1):
    ints, coefs = prova(cco2, alt1, alt2)

    nlte_corr[(cco2, 'c')] = np.array(ints)
    nlte_corr[(cco2, 'm1')] = np.array([co[0] for co in coefs])
    nlte_corr[(cco2, 'm2')] = np.array([co[1] for co in coefs])
    nlte_corr[(cco2, 'm3')] = np.array([co[2] for co in coefs])
    nlte_corr[(cco2, 'm4')] = np.array([co[3] for co in coefs])

pickle.dump(nlte_corr, open(cart_out_rep + 'nlte_corr_low.p', 'wb'))
### ora il check

# ### OK! now I use the dotprods and the regrcoef to reconstruct the a and b coeffs, and compute the hr and check wrt the reference and the varfit5.
cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'
tot_coeff_co2 = pickle.load(open(cart_out_2 + 'tot_coeffs_co2_NLTE.p', 'rb'))

colors = npl.color_set(5)
colors = colors[:3] + [colors[4]]

figs = []
figs2 = []
a0s = []
a1s = []
for cco2 in range(1,npl.n_co2prof+1):
    fig2, ax2 = plt.subplots()
    for i, (atm, col) in enumerate(zip(allatms, npl.color_set(6))):
        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]

        acoeff, bcoeff, asurf, bsurf = npl.coeffs_from_eofreg(cco2, temp, surf_temp, method = '1eof', regrcoef = regrcoef)
        hr_new = npl.hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = npl.n_alts_all)

        hra, hrb = npl.hr_from_ab_diagnondiag(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts=npl.n_alts_all)
        hr_nlte_corr = nlte_corr[(cco2, 'c')] + nlte_corr[(cco2, 'm1')] * hra[alt1:alt2] + nlte_corr[(cco2, 'm2')] * hrb[alt1:alt2] + nlte_corr[(cco2, 'm3')] * x0[i] + nlte_corr[(cco2, 'm4')] * x1[i]

        acoeff, bcoeff, asurf, bsurf = npl.coeffs_from_eofreg(cco2, temp, surf_temp, method = '2eof', regrcoef = regrcoef)
        hr_new_v2 = npl.hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = npl.n_alts_all)
        hra, hrb = npl.hr_from_ab_diagnondiag(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts=npl.n_alts_all)
        hr_nlte_corr = nlte_corr[(cco2, 'c')] + nlte_corr[(cco2, 'm1')] * hra[alt1:alt2] + nlte_corr[(cco2, 'm2')] * hrb[alt1:alt2] + nlte_corr[(cco2, 'm3')] * x0[i] + nlte_corr[(cco2, 'm4')] * x1[i]

        hr_new_v2[alt1:alt2] = hr_new_v2[alt1:alt2] + hr_nlte_corr
        ax2.plot(hr_nlte_corr, np.arange(alt1, alt2), color = col, linestyle = '--')

        tip = 'varfit5_nlte'
        acoeff = tot_coeff_co2[(tip, 'acoeff', cco2)]
        bcoeff = tot_coeff_co2[(tip, 'bcoeff', cco2)]
        asurf = tot_coeff_co2[(tip, 'asurf', cco2)]
        bsurf = tot_coeff_co2[(tip, 'bsurf', cco2)]

        hr_vf5 = npl.hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = 51)
        #hr_vf5 = np.concatenate([hr_vf5, np.nan*np.ones(15)])

        hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_ref')]

        ax2.plot(hr_ref[alt1:alt2]-all_coeffs_nlte[(atm, cco2, 'hr_lte')][alt1:alt2], np.arange(alt1, alt2), color = col, label = atm)

        nmax = 51
        labels = ['ref', 'veof', 'veof2', 'vf5']
        hrs = [hr_ref[:nmax], hr_new[:nmax], hr_new_v2[:nmax], hr_vf5[:nmax]]
        tit = 'co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'index'
        #labels = ['ref'] + alltips + ['fomi rescale (no fit)', 'old param']
        fig, a0, a1 = npl.manuel_plot(np.arange(nmax), hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-2, 2), xlim = (-40, 10), ylim = (0, 51), linestyles = ['-', '--', '--', ':'], colors = colors, orizlines = [39, 50])

        figs.append(fig)
        a0s.append(a0)
        a1s.append(a1)

    ax2.set_title(str(cco2))
    figs2.append(fig2)

    npl.adjust_ax_scale(a0s)
    npl.adjust_ax_scale(a1s)

npl.plot_pdfpages(cart_out_rep + 'check_reparam_NLTE_low.pdf', figs)
npl.plot_pdfpages(cart_out_rep + 'check_reparam_NLTEcorrection.pdf', figs2)
