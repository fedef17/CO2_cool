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
    cart_out_2 = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE/'

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

tot_coeff_co2_old = pickle.load(open(cart_out + 'tot_coeffs_co2_v1_LTE.p', 'r'))
tot_coeff_co2 = dict()

varfit_xis = pickle.load(open(cart_out+'varfit_LTE_v2b.p', 'rb'))
varfit_xis_2 = pickle.load(open(cart_out+'varfit_LTE_v3b.p', 'rb'))

varfit_xis_4 = pickle.load(open(cart_out+'varfit_LTE_v4.p', 'rb'))
varfit_xis_5 = pickle.load(open(cart_out+'varfit_LTE_v5.p', 'rb'))

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

    # PATCH: the old version only had 6 co2 profiles, setting the lowest profile to nan
    for cotip in ['unifit', 'varfit']:
        for conam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
            if cco2 == 1:
                tot_coeff_co2[(cotip, conam, cco2)] = np.nan * tot_coeff_co2_old[(cotip, conam, 1)]
            else:
                tot_coeff_co2[(cotip, conam, cco2)] = tot_coeff_co2_old[(cotip, conam, cco2-1)]

    acoeff, bcoeff, asurf, bsurf = npl.ab_from_xi_abfit(varfit_xis_4, cco2)
    tot_coeff_co2[('varfit4', 'acoeff', cco2)] = acoeff
    tot_coeff_co2[('varfit4', 'bcoeff', cco2)] = bcoeff
    tot_coeff_co2[('varfit4', 'asurf', cco2)] = asurf
    tot_coeff_co2[('varfit4', 'bsurf', cco2)] = bsurf

    acoeff, bcoeff, asurf, bsurf = npl.ab_from_xi_abfit(varfit_xis_5, cco2)
    tot_coeff_co2[('varfit5', 'acoeff', cco2)] = acoeff
    tot_coeff_co2[('varfit5', 'bcoeff', cco2)] = bcoeff
    tot_coeff_co2[('varfit5', 'asurf', cco2)] = asurf
    tot_coeff_co2[('varfit5', 'bsurf', cco2)] = bsurf

pickle.dump(tot_coeff_co2, open(cart_out + 'tot_coeffs_co2_v2_LTE.p', 'wb'))
tot_coeff_co2 = pickle.load(open(cart_out + 'tot_coeffs_co2_v2_LTE.p', 'rb'))

# poi fai un check please con npl.coeff_from_interp_log()
figs = []
figs2 = []
a0s = []
a1s = []

co2profs = [atm_pt[('mle', cco2, 'co2')] for cco2 in range(1,8)]

fit_score = dict()
alltips = ['unifit', 'varfit', 'varfit2', 'varfit3', 'varfit4', 'varfit5']
for tip in alltips:
    for sco in ['lte', 'lte+trans']:
        for cos in ['std', 'max']:
            fit_score[(tip, sco, cos)] = []

for cco2 in range(1,8):
    co2pr = co2profs[cco2-1]

    for atm in allatms:
        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]

        tit = 'co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'Alt (km)'

        hr_ref = all_coeffs[(atm, cco2, 'hr_ref')][:n_alts]

        hr_calcs = []
        for tip in ['unifit', 'varfit', 'varfit2', 'varfit3', 'varfit4', 'varfit5']:
            acoeff_cco2 = tot_coeff_co2[(tip, 'acoeff', cco2)]
            bcoeff_cco2 = tot_coeff_co2[(tip, 'bcoeff', cco2)]
            asurf_cco2 = tot_coeff_co2[(tip, 'asurf', cco2)]
            bsurf_cco2 = tot_coeff_co2[(tip, 'bsurf', cco2)]

            hr_calc = npl.hr_from_ab(acoeff_cco2, bcoeff_cco2, asurf_cco2, bsurf_cco2, temp, surf_temp)[:n_alts]
            hr_calcs.append(hr_calc)
            fit_score[(tip, 'lte', 'std')].append(np.sqrt(np.mean((hr_calc[:n_alts_lte]-hr_ref[:n_alts_lte])**2)))
            fit_score[(tip, 'lte+trans', 'std')].append(np.sqrt(np.mean((hr_calc-hr_ref)**2)))

            fit_score[(tip, 'lte', 'max')].append(np.max(np.abs(hr_calc[:n_alts_lte]-hr_ref[:n_alts_lte])))
            fit_score[(tip, 'lte+trans', 'max')].append(np.max(np.abs(hr_calc-hr_ref)))

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
        labels = ['ref', 'unifit', 'varfit', 'varfit2', 'varfit3', 'varfit4', 'varfit5']
        fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-2.5, 2.5))

        figs2.append(fig)
        a0s.append(a0)
        a1s.append(a1)

npl.adjust_ax_scale(a0s)
npl.adjust_ax_scale(a1s)
# npl.plot_pdfpages(cart_out + 'check_newparam_LTE_final_newvsold.pdf', figs)

npl.plot_pdfpages(cart_out + 'check_newparam_LTE_final_LEASTSQUARES_v3_abfit.pdf', figs2)

for cos in ['std', 'max']:
    if cos == 'std':
        print('Average stddev of param in region.\n')
    else:
        print('Max absolute error of param in region.\n')

    for sco in ['lte', 'lte+trans']:
        print('---------------- \n')
        print(sco + ' region \n')
        allsco = []
        for tip in alltips:
            if cos == 'std':
                allsco.append(np.nanmean(fit_score[(tip, sco, cos)]))
                print('{} {}: {:6.3f} K'.format(tip, sco, allsco[-1]))
            else:
                allsco.append(np.nanmax(fit_score[(tip, sco, cos)]))
                print('{} {}: {:6.3f} K'.format(tip, sco, allsco[-1]))

        print('BEST TIP: {} \n'.format(alltips[np.argmin(allsco)]))

        allsco = []
        for tip in alltips:
            if cos == 'std':
                allsco.append(np.nanmean(fit_score[(tip, sco, cos)][-2:]))
                print('{} {}: {:6.3f} K'.format(tip, sco, allsco[-1]))
            else:
                allsco.append(np.nanmax(fit_score[(tip, sco, cos)][-2:]))
                print('{} {}: {:6.3f} K'.format(tip, sco, allsco[-1]))

        print('BEST TIP FOR SAS and SAW: {} \n'.format(alltips[np.argmin(allsco)]))


for cco2 in range(1,8):
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
