#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
from matplotlib import cm
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

import newparam_lib as npl

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

n_alts = 54

all_alts = atm_pt[('mle', 'alts')]
alts = atm_pt[('mle', 'alts')][:n_alts]

n_alts_lte = 40

figs = []
a0s = []
a1s = []

tot_coeff_co2 = pickle.load(open(cart_out + 'tot_coeffs_co2_v1_LTE.p', 'r'))

varfit_xis = pickle.load(open(cart_out+'varfit_LTE_v2b.p', 'rb'))
varfit_xis_2 = pickle.load(open(cart_out+'varfit_LTE_v3b.p', 'rb'))

for cco2 in allco2:
    acoeff, bcoeff = npl.ab_from_xi_varfit(varfit_xis, cco2)
    asurf, bsurf = npl.absurf_from_xi_varfit(varfit_xis, cco2)

    tot_coeff_co2[('varfit2', 'acoeff', cco2)] = acoeff
    tot_coeff_co2[('varfit2', 'bcoeff', cco2)] = bcoeff
    tot_coeff_co2[('varfit2', 'asurf', cco2)] = asurf
    tot_coeff_co2[('varfit2', 'bsurf', cco2)] = bsurf

    acoeff, bcoeff = npl.ab_from_xi_varfit(varfit_xis_2, cco2)
    asurf, bsurf = npl.absurf_from_xi_varfit(varfit_xis_2, cco2)

    tot_coeff_co2[('varfit3', 'acoeff', cco2)] = acoeff
    tot_coeff_co2[('varfit3', 'bcoeff', cco2)] = bcoeff
    tot_coeff_co2[('varfit3', 'asurf', cco2)] = asurf
    tot_coeff_co2[('varfit3', 'bsurf', cco2)] = bsurf

pickle.dump(tot_coeff_co2, open(cart_out + 'tot_coeffs_co2_v2_LTE.p', 'wb'))
tot_coeff_co2 = pickle.load(open(cart_out + 'tot_coeffs_co2_v2_LTE.p', 'rb'))

# poi fai un check please con npl.coeff_from_interp()
figs = []
figs2 = []
a0s = []
a1s = []

co2profs = [atm_pt[('mle', cco2, 'co2')] for cco2 in range(1,7)]

for cco2 in range(1,7):
    co2pr = co2profs[cco2-1]

    for atm in allatms:
        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]

        tit = 'co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'Alt (km)'

        hr_ref = all_coeffs[(atm, cco2, 'hr_ref')][:n_alts]

        hr_calcs = []
        for tip in ['unifit', 'varfit', 'varfit2', 'varfit3']:
            acoeff_cco2 = tot_coeff_co2[(tip, 'acoeff', cco2)]
            bcoeff_cco2 = tot_coeff_co2[(tip, 'bcoeff', cco2)]
            asurf_cco2 = tot_coeff_co2[(tip, 'asurf', cco2)]
            bsurf_cco2 = tot_coeff_co2[(tip, 'bsurf', cco2)]

            hr_calc = npl.hr_from_ab(acoeff_cco2, bcoeff_cco2, asurf_cco2, bsurf_cco2, temp, surf_temp)[:n_alts]
            hr_calcs.append(hr_calc)

        # pres = atm_pt[(atm, 'pres')]
        # print(np.median(co2pr))
        # alt_fomi, hr_fomi = npl.old_param(all_alts, temp, pres, co2pr)
        # oldco = spline(alt_fomi, hr_fomi)
        # hr_fomi = oldco(alts)
        #
        # hr_calc = npl.new_param_LTE(interp_coeffs, temp, co2pr, surf_temp = surf_temp)[:n_alts]
        #
        # hrs = [hr_ref, hr_fomi, hr_calc]
        # labels = ['ref', 'old_param', 'new_param']
        # colors = ['black', 'steelblue', 'orange']
        # fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-2.5, 2.5), colors = colors)
        #
        # figs.append(fig)
        # a0s.append(a0)
        # a1s.append(a1)

        hrs = [hr_ref] + hr_calcs
        labels = ['ref', 'unifit', 'varfit', 'varfit2', 'varfit3']
        fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-2.5, 2.5))

        figs2.append(fig)
        a0s.append(a0)
        a1s.append(a1)

npl.adjust_ax_scale(a0s)
npl.adjust_ax_scale(a1s)
# npl.plot_pdfpages(cart_out + 'check_newparam_LTE_final_newvsold.pdf', figs)

npl.plot_pdfpages(cart_out + 'check_newparam_LTE_final_LEASTSQUARES_v2.pdf', figs2)


for cco2 in range(1,7):
    co2pr = co2profs[cco2-1]

    for atm in allatms:
        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]

        tit = 'co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'Alt (km)'

        hr_ref = all_coeffs[(atm, cco2, 'hr_ref')][:n_alts]

        hr_calcs = []
        for atm in allatms:
            acoeff_cco2 = all_coeffs[(atm, cco2, 'acoeff')]
            bcoeff_cco2 = all_coeffs[(atm, cco2, 'bcoeff')]
            asurf_cco2 = all_coeffs[(atm, cco2, 'asurf')]
            bsurf_cco2 = all_coeffs[(atm, cco2, 'bsurf')]

            hr_calc = npl.hr_from_ab(acoeff_cco2, bcoeff_cco2, asurf_cco2, bsurf_cco2, temp, surf_temp)[:n_alts]
            hr_calcs.append(hr_calc)

        hrs = [hr_ref] + hr_calcs
        labels = ['ref'] + allatms
        fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-2.5, 2.5))

        figs2.append(fig)
        a0s.append(a0)
        a1s.append(a1)

npl.adjust_ax_scale(a0s)
npl.adjust_ax_scale(a1s)
# npl.plot_pdfpages(cart_out + 'check_newparam_LTE_final_newvsold.pdf', figs)

npl.plot_pdfpages(cart_out + 'check_newparam_LTE_final_ATMCOEFFS.pdf', figs2)
