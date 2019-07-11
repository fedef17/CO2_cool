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

# sys.path.insert(0, '/home/fedefab/Scrivania/Research/Dotto/Git/spect_robot/')
# sys.path.insert(0, '/home/fedefab/Scrivania/Research/Dotto/Git/pythall/')
sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
import spect_base_module as sbm
import spect_classes as spcl

import newparam_lib as npl

##############################################################
kbc = const.k/(const.h*100*const.c) # 0.69503
kboltz = 1.38064853e-23 # J/K
E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition

cp = 1.005e7 # specific enthalpy dry air - erg g-1 K-1
#############################################################

# cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'
cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'

allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweigths = dict(zip(allatms, atmweigths))
allco2 = np.arange(1,7)

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v1.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt.p'))

n_alts = 55

best_unif = pickle.load(open(cart_out+'best_uniform_allco2.p'))
best_unif_v2 = pickle.load(open(cart_out+'best_uniform_allco2_v2.p'))
best_var = pickle.load(open(cart_out+'best_atx0_allco2.p'))

best_var.update(pickle.load(open(cart_out+'best_atx0_allco2_High.p')))

alts = atm_pt[('mle', 'alts')][:n_alts]

n_alts_lte = 40

figs = []
a0s = []
a1s = []


tot_coeff_co2 = dict()

for cco2 in allco2:
    xis_unif = best_unif[cco2]#/np.sum(best_unif[cco2])
    xis2_unif = best_unif_v2[cco2]#/np.sum(best_unif_v2[cco2])
    #print(cco2, xis)

    acoeff, bcoeff = npl.ab_from_xi_unifit(xis2_unif, cco2)
    asurf, bsurf = npl.absurf_from_xi_unifit(xis2_unif, cco2)

    tot_coeff_co2[('unifit', 'acoeff', cco2)] = acoeff
    tot_coeff_co2[('unifit', 'bcoeff', cco2)] = bcoeff
    tot_coeff_co2[('unifit', 'asurf', cco2)] = asurf
    tot_coeff_co2[('unifit', 'bsurf', cco2)] = bsurf

    acoeff, bcoeff = npl.ab_from_xi_varfit(best_var, cco2)
    asurf, bsurf = npl.absurf_from_xi_varfit(best_var, cco2)

    tot_coeff_co2[('varfit', 'acoeff', cco2)] = acoeff
    tot_coeff_co2[('varfit', 'bcoeff', cco2)] = bcoeff
    tot_coeff_co2[('varfit', 'asurf', cco2)] = asurf
    tot_coeff_co2[('varfit', 'bsurf', cco2)] = bsurf


# STEP 5
# ORA devo interpolare ad ogni x0 log(a(x0)/cco2(x0)), log(b/cco2)

interp_coeffs = dict()
for tip in ['unifit', 'varfit']:

    co2profs = [atm_pt[('mle', cco2, 'co2')] for cco2 in range(1,7)]

    for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        coeffs = [tot_coeff_co2[(tip, nam, cco2)] for cco2 in range(1,7)]

        mcoeff, ccoeff, signc, errm, errc = npl.interp_coeff_logco2(coeffs, co2profs)
        interp_coeffs[(tip, nam, 'mcoeff')] = mcoeff
        interp_coeffs[(tip, nam, 'ccoeff')] = ccoeff
        interp_coeffs[(tip, nam, 'signc')] = signc
        interp_coeffs[(tip, nam, 'err_mcoeff')] = errm
        interp_coeffs[(tip, nam, 'err_ccoeff')] = errc

    ###########################################################################

    n_alts = 55
    plt.ion()

    cco2 = 3
    co2pr = co2profs[cco2-1]

    for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        mc = interp_coeffs[(tip, nam, 'mcoeff')]
        cc = interp_coeffs[(tip, nam, 'ccoeff')]
        sc = interp_coeffs[(tip, nam, 'signc')]
        coeff = npl.coeff_from_interp(mc, cc, sc, co2pr)

        # i coeff specifici per ogni cco2:
        coeff_cco2 = tot_coeff_co2[(tip, nam, cco2)]

        fig = plt.figure()
        plt.title(nam + ' - cco2: {}'.format(cco2))
        if coeff.ndim == 2:
            plt.plot(coeff[:,10][:n_alts], alts, label = 'step 5')
            plt.plot(coeff_cco2[:,10][:n_alts], alts, label = 'step 4')
        else:
            plt.plot(coeff[:n_alts], alts, label = 'step 5')
            plt.plot(coeff_cco2[:n_alts], alts, label = 'step 4')

        plt.xlabel('coeff')
        plt.ylabel('alt (km)')
        plt.grid()
        fig.savefig(cart_out + 'coeff_step5_{}_{}.pdf'.format(nam, tip))

    plt.ioff()

    # poi fai un check please con npl.coeff_from_interp()
    figs = []
    a0s = []
    a1s = []

    for cco2 in range(1,7):
        co2pr = co2profs[cco2-1]

        # i coeffs universali:
        coeffs = []
        for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
            mc = interp_coeffs[(tip, nam, 'mcoeff')]
            cc = interp_coeffs[(tip, nam, 'ccoeff')]
            sc = interp_coeffs[(tip, nam, 'signc')]
            coeffs.append(npl.coeff_from_interp(mc, cc, sc, co2pr))

        acoeff, bcoeff, asurf, bsurf = coeffs

        # i coeff specifici per ogni cco2:
        acoeff_cco2 = tot_coeff_co2[(tip, 'acoeff', cco2)]
        bcoeff_cco2 = tot_coeff_co2[(tip, 'bcoeff', cco2)]
        asurf_cco2 = tot_coeff_co2[(tip, 'asurf', cco2)]
        bsurf_cco2 = tot_coeff_co2[(tip, 'bsurf', cco2)]

        for atm in allatms:
            temp = atm_pt[(atm, 'temp')]
            surf_temp = atm_pt[(atm, 'surf_temp')]

            tit = 'co2: {} - atm: {}'.format(cco2, atm)
            xlab = 'CR (K/day)'
            ylab = 'Alt (km)'


            hr_ref = all_coeffs[(atm, cco2, 'hr_ref')][:n_alts]
            hr_calc_step5 = npl.hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)[:n_alts]
            hr_calc_step4 = npl.hr_from_ab(acoeff_cco2, bcoeff_cco2, asurf_cco2, bsurf_cco2, temp, surf_temp)[:n_alts]

            hrs = [hr_ref, hr_calc_step4, hr_calc_step5]
            labels = ['ref', 'step 4', 'step 5']
            fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-2.5, 2.5))

            figs.append(fig)
            a0s.append(a0)
            a1s.append(a1)

    npl.adjust_ax_scale(a0s)
    npl.adjust_ax_scale(a1s)
    npl.plot_pdfpages(cart_out + 'check_newparam_LTE_final_{}.pdf'.format(tip), figs)

# poi fai un check please con npl.coeff_from_interp()
figs = []
a0s = []
a1s = []

for cco2 in range(1,7):
    co2pr = co2profs[cco2-1]

    # i coeffs universali:
    tip = 'unifit'
    coeffs = []
    for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        mc = interp_coeffs[(tip, nam, 'mcoeff')]
        cc = interp_coeffs[(tip, nam, 'ccoeff')]
        sc = interp_coeffs[(tip, nam, 'signc')]
        coeffs.append(npl.coeff_from_interp(mc, cc, sc, co2pr))

    acoeff, bcoeff, asurf, bsurf = coeffs

    tip = 'varfit'
    coeffs = []
    for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        mc = interp_coeffs[(tip, nam, 'mcoeff')]
        cc = interp_coeffs[(tip, nam, 'ccoeff')]
        sc = interp_coeffs[(tip, nam, 'signc')]
        coeffs.append(npl.coeff_from_interp(mc, cc, sc, co2pr))

    acoeffv, bcoeffv, asurfv, bsurfv = coeffs

    for atm in allatms:
        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]

        tit = 'co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'Alt (km)'


        hr_ref = all_coeffs[(atm, cco2, 'hr_ref')][:n_alts]
        hr_calc_uni = npl.hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)[:n_alts]
        hr_calc_var = npl.hr_from_ab(acoeffv, bcoeffv, asurfv, bsurfv, temp, surf_temp)[:n_alts]

        hrs = [hr_ref, hr_calc_uni, hr_calc_var]
        labels = ['ref', 'unifit', 'varfit']
        fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-2.5, 2.5))

        figs.append(fig)
        a0s.append(a0)
        a1s.append(a1)

npl.adjust_ax_scale(a0s)
npl.adjust_ax_scale(a1s)
npl.plot_pdfpages(cart_out + 'check_newparam_LTE_final_univsvar.pdf', figs)

pickle.dump(interp_coeffs, open(cart_out + 'interp_coeffs_v1_LTE.p', 'w'))
pickle.dump(tot_coeff_co2, open(cart_out + 'tot_coeffs_co2_v1_LTE.p', 'w'))

#### Voglio vedere come si comporta a per diversi cco2
plt.close('all')
plt.ion()

fig = plt.figure()
for cco2 in range(1,7):
    acoeff = tot_coeff_co2[('varfit', 'acoeff', cco2)]

    plt.plot(abs(acoeff[:, 10][:n_alts]), alts, label = str(cco2))
plt.xscale('log')


##### Voglio vedere se a è legato alla quota della tropopausa
fig = plt.figure()
cco2 = 3

stratops = []
for atm in allatms:
    temp = atm_pt[(atm, 'temp')]
    stratops.append(alts[np.argmin(temp[:30])])

ord = np.argsort(stratops)
allatms_ord = np.array(allatms)[ord]
stratops = np.array(stratops)[ord]

stratops_norm = (np.array(stratops)-np.min(stratops))/(np.max(stratops)-np.min(stratops))

colors = npl.color_set(len(allatms))
for atm, strp, col in zip(allatms_ord, stratops, colors):
    acoeff = all_coeffs[(atm, cco2, 'acoeff')]

    plt.plot(abs(acoeff[:, 10][:n_alts]), alts, color = col, label = '{} km'.format(strp))
plt.xscale('log')

fig = plt.figure()
for i, (strp, col) in enumerate(zip(stratops, colors)):
    plt.scatter(i, strp, color = col, s = 10)
