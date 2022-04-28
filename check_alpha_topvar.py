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
n_alts_cs = 80
ntops = [60, 65, 70]

aunifs = dict()
asingle = dict()
afit = dict()
for n_top in ntops:
    aunifs[n_top], asingle[('v8', n_top)] = pickle.load(open(cart_out_rep +  'alpha_singleatm_v2_top{}.p'.format(n_top), 'rb'))
    _, asingle[('v9', n_top)] = pickle.load(open(cart_out_rep + 'alpha_singleatm_v3i_top{}.p'.format(n_top), 'rb'))

    afit[('v8', n_top)] = pickle.load(open(cart_out_rep + 'alpha_fit_nl0_v8_top{}.p'.format(n_top), 'rb'))
    afit[('v9', n_top)] = pickle.load(open(cart_out_rep + 'alpha_fit_nl0_9i_top{}.p'.format(n_top), 'rb'))

# OK.

############################################################
tot_coeff_co2 = pickle.load(open(cart_out_2 + 'tot_coeffs_co2_NLTE.p', 'rb'))

figs_fit = []
figs_sing = []

a0s = []
a1s = []
for cco2 in range(1, npl.n_co2prof+1):
    for ii, atm in enumerate(allatms):
        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]
        pres = atm_pt[(atm, 'pres')]

        co2vmr = atm_pt[(atm, cco2, 'co2')]
        ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
        o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
        n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]

        L_esc = cose_upper_atm[(atm, cco2, 'L_esc_all')]
        lamb = cose_upper_atm[(atm, cco2, 'lamb')]
        #MM = cose_upper_atm[(atm, cco2, 'MM')]
        MM = npl.calc_MM(ovmr, o2vmr, n2vmr)

        hr_calc = npl.hr_reparam_low(cco2, temp, surf_temp, regrcoef = regrcoef, nlte_corr = nlte_corr)

        hr_sing = []
        hr_fit = []
        labs = []
        cols = []
        lsti = []
        for lst, tip in zip(['--', ':'], ['v8', 'v9']):
            if tip == 'v9':
                continue
            for col, n_top in zip(['forestgreen','orange','indianred'], ntops):
                alpha_fit = afit[(tip, n_top)]
                alphaok = asingle[(tip, n_top)][cco2][allatms.index(atm), :]

                alpha_nl = npl.alpha_from_fit(temp, surf_temp, lamb, alpha_fit[cco2], alpha_max = alpha_fit[('max', cco2)], alpha_min = alpha_fit[('min', cco2)], alpha_cose = alpha_fit, alt2 = alt2, n_top = n_top, method = 'nl0')

                hr_calc_nl = npl.recformula(alpha_nl, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, ovmr = ovmr, n_alts_cs = n_alts_cs)
                hr_fit.append(hr_calc_nl)
                labs.append('{} top{}'.format(tip, n_top))

                hr_calc_aok = npl.recformula(alphaok, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, ovmr = ovmr, n_alts_cs = n_alts_cs)
                hr_sing.append(hr_calc_aok)
                #labs.append('{} top{}'.format(tip, n_top))

                lsti.append(lst)
                cols.append(col)

                #hr_calc_aunif = npl.recformula(alpha_unif[cco2-1, :], L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, ovmr = ovmr, n_alts_cs = n_alts_cs)

        hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_ref')]

        # alt_fomi, hr_fomi = npl.old_param(all_alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr)
        # oldco = spline(alt_fomi, hr_fomi)
        # hr_fomi = oldco(alts)

        tit = 'FIT. co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'index'
        hrs = [hr_ref] + hr_fit
        labels = ['ref'] + labs
        colors = ['steelblue'] + cols
        linestyles = ['-'] + lsti

        fig, a0, a1 = npl.manuel_plot(np.arange(npl.n_alts_all), hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-25, 25), xlim = (-1000, 10), ylim = (40, 80), linestyles = linestyles, colors = colors, orizlines = [40, alt2, n_top], linewidth = 2.)

        figs_fit.append(fig)
        # a0s.append(a0)
        # a1s.append(a1)
        #
        # npl.adjust_ax_scale(a0s)
        # npl.adjust_ax_scale(a1s)

        tit = 'SING. co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'index'
        hrs = [hr_ref] + hr_sing
        labels = ['ref'] + labs
        colors = ['steelblue'] + cols
        linestyles = ['-'] + lsti

        fig, a0, a1 = npl.manuel_plot(np.arange(npl.n_alts_all), hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-25, 25), xlim = (-1000, 10), ylim = (40, 80), linestyles = linestyles, colors = colors, orizlines = [40, alt2, n_top], linewidth = 2.)

        figs_sing.append(fig)

npl.plot_pdfpages(cart_out_F + 'check_alpha_topvar_fit.pdf', figs_fit)
npl.plot_pdfpages(cart_out_F + 'check_alpha_topvar_sing.pdf', figs_sing)
# npl.plot_pdfpages(cart_out_F + 'check_alpha_topvar_fit_v8only.pdf', figs_fit)
# npl.plot_pdfpages(cart_out_F + 'check_alpha_topvar_sing_v8only.pdf', figs_sing)
