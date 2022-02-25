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

import spect_base_module as sbm

cart_in = cart_base + '../MIPAS_2009/'

cart_out = cart_base + 'LTE/'
cart_out_2 = cart_base + 'NLTE/'
cart_out_rep = cart_base + 'NLTE_reparam/'
cart_out_3 = cart_base + 'NLTE_upper/'

cart_out_F = cart_base + 'newpar_allatm_2/'

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

#crun = '/home/fabiano/Research/lavori/CO2_cooling/cart_run_fomi/'
crun = '/home/fedef/Research/lavori/CO2_cooling/cart_run_fomi/'
################################################################################
cart_out_mip = cart_base + 'mipas_check/'

all_alts = atm_pt[('mle', 'alts')]
interp_coeffs = npl.precalc_interp()

figs = []
a0s = []
a1s = []
for cco2 in range(1, npl.n_co2prof+1):
    for ii, atm in enumerate(allatms):
        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]
        pres = atm_pt[(atm, 'pres')]

        hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_ref')]

        x_ref = np.log(1000./pres)
        #x_ref = np.arange(0.125, np.max(x) + 0.001, 0.25)

        co2vmr = atm_pt[(atm, cco2, 'co2')]
        ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
        o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
        n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]

        L_esc = cose_upper_atm[(atm, cco2, 'L_esc_all_wutop')]
        lamb = cose_upper_atm[(atm, cco2, 'lamb')]
        MM = cose_upper_atm[(atm, cco2, 'MM')]

        alt_fomi, x_fomi, cr_fomi = npl.old_param(all_alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr, input_in_ppm = False, cart_run_fomi = crun)
        spl = spline(x_fomi, cr_fomi)
        crok = spl(x_ref)

        # i0 = 49
        i0 = 50

        # Loading exactly fomi alpha and L_esc
        zunk = np.loadtxt(crun + 'debug_alpha__mipas.dat')
        X_fom = zunk[:, 1]
        spl = spline(X_fom, np.exp(zunk[:,3]))
        realpha = spl(x_ref[i0:i0+6])
        print(cco2, realpha)
        alp = np.append(realpha, np.ones(9))

        cr_fomialpha = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug_alpha = alp)

        ali = np.exp(zunk[:,4]) # with no correction
        spl = spline(X_fom, ali)
        reLesc = spl(x_ref[i0:i0+17])
        reL = np.zeros(len(L_esc))
        reL[i0:i0+17] = reLesc
        reL[i0+17:] = 1.

        if len(reL) >= len(x_ref):
            relok = reL[:len(x_ref)]
        else:
            relok = np.append(reL, np.ones(len(x_ref)-len(reL)))

        cr_fa_fL = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug_alpha = alp, debug_Lesc = relok)

        ####### ora faccio un fa_fL ma partendo esattamente da fomi a x = 12.5
        cr_fa_fL_start = npl.recformula(alp[:6], relok, lamb, crok, co2vmr, MM, temp, n_alts_trlo = i0+1, n_alts_trhi = i0+6, n_alts_cs = 65, ovmr = ovmr)


        cr_new = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs)

        cr_new_oldfactor = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, factor_from_code = False)

        cr_fa_fL_start_new = npl.recformula(alp[:6], relok, lamb, cr_new, co2vmr, MM, temp, n_alts_trlo = i0+1, n_alts_trhi = i0+6, n_alts_cs = 65, ovmr = ovmr)


        tit = 'co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'index'

        # labels = ['nlte_ref', 'pop_nl0_wt', 'fomialpha', 'fa_fL' , 'old']#, 'fomi']
        # hrs = [hr_ref, cr_new, cr_fomialpha, cr_fa_fL, crok]
        # colors = ['violet', 'red', 'orange', 'forestgreen', 'blue']

        # fa_fL e fa_fL_start_new sono identici!

        labels = ['nlte_ref', 'pop_nl0_wt', 'fomialpha', 'old', 'fa_fL_start', 'fa_fL_start_new', 'new_oldfact']#, 'fomi']
        hrs = [hr_ref, cr_new, cr_fomialpha, crok, cr_fa_fL_start, cr_fa_fL_start_new, cr_new_oldfactor]
        colors = ['violet', 'red', 'orange', 'blue', 'forestgreen', 'brown', 'grey']

        fig, a0, a1 = npl.manuel_plot(np.arange(npl.n_alts_all), hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-15, 15), xlim = (-70, 10), ylim = (40, 67), linestyles = ['-', '--', ':', '-', ':', ':', ':'], colors = colors, orizlines = [40, 50])

        figs.append(fig)
        a0s.append(a0)
        a1s.append(a1)

        npl.adjust_ax_scale(a0s)
        npl.adjust_ax_scale(a1s)

npl.plot_pdfpages(cart_out_mip + 'check_fomialpha_refatm.pdf', figs)
