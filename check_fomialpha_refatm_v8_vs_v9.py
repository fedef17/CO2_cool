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

cart_out_F = cart_base + 'newpar_allatm_v2/'

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

regrcoef = pickle.load(open(cart_out_rep + 'regrcoef_v3.p', 'rb'))
nlte_corr = pickle.load(open(cart_out_rep + 'nlte_corr_low.p', 'rb'))

crun = cart_base + '../cart_run_fomi/'
################################################################################
alt2 = 51
n_alts_cs = 80

allntops = [60, 63, 65, 67, 70]
#n_top = 65 #!!!! IMPORTANT!
alpha_fit_all = dict()
alpha_unif = dict()
for n_top in allntops:
    alpha_unif[n_top], _ = pickle.load(open(cart_out_rep +     'alpha_singleatm_v2_top{}.p'.format(n_top), 'rb'))
    # _, alpha_dic_atm[('v9', n_top)] = pickle.load(open(cart_out_rep + 'alpha_singleatm_v3_inverse_top{}.p'.format(n_top), 'rb'))
    for tip in ['v8', '9i', 'v10']:
        for strat in ['4e', 'nl0']:
            alpha_fit_all[(tip, strat, n_top)] = pickle.load(open(cart_out_rep +'alpha_fit_{}_{}_top{}.p'.format(strat, tip, n_top), 'rb'))

# figs = []
# axs = []
# for cco2 in range(1, 9):
#     fig, ax = plt.subplots()
#     for atm in allatms:
#         ax.plot(alpha_dic_atm[cco2][allatms.index(atm)], np.arange(10), label = atm)
#     ax.legend()
#     ax.grid()
#     ax.set_title('co2: '+str(cco2))
#     ax.set_xlabel('alpha')
#     axs.append(ax)
#     figs.append(fig)
#
# npl.adjust_ax_scale(axs)
# npl.plot_pdfpages(cart_out_F + 'check_all_alphas_refatm.pdf', figs)

cart_out_mip = cart_base + 'mipas_check/'

resfile = cart_out_F + 'results_best_strategy.txt'
filo = open(resfile, 'w')

all_alts = atm_pt[('mle', 'alts')]

interp_coeffs = npl.precalc_interp(coeff_tag = '{}-{}-{}'.format('v10', 'nl0', 65))

interp_coeffs_old = npl.precalc_interp(n_top = 65, coeff_file = cart_base + 'reparam_allatm/coeffs_finale_oldv10.p')

# tip in v8, v9, compare
def calc_all_plot(cco2, atm, tip, strat, compare = None, debug = True):
    ii = allatms.index(atm)

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

    L_esc = cose_upper_atm[(atm, cco2, 'L_esc_all_extP')]
    lamb = cose_upper_atm[(atm, cco2, 'lamb')]
    MM = cose_upper_atm[(atm, cco2, 'MM')]

    alt_fomi, x_fomi, cr_fomi = npl.old_param(all_alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr, input_in_ppm = False, cart_run_fomi = crun)
    spl = spline(x_fomi, cr_fomi)
    crok = spl(x_ref)

    # # i0 = 49
    # i0 = 50
    #
    # # Loading exactly fomi alpha and L_esc
    # zunk = np.loadtxt(crun + 'debug_alpha__mipas.dat')
    # X_fom = zunk[:, 1]
    # spl = spline(X_fom, np.exp(zunk[:,3]))
    # realpha = spl(x_ref[i0:i0+6])
    # print(cco2, realpha)
    # alp = np.append(realpha, np.ones(9))
    #
    # #cr_fomialpha = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug_alpha = alp)
    #
    # ali = np.exp(zunk[:,4]) # with no correction
    # spl = spline(X_fom, ali)
    # reLesc = spl(x_ref[i0:i0+17])
    # reL = np.zeros(len(L_esc))
    # reL[i0:i0+17] = reLesc
    # reL[i0+17:] = 1.
    #
    # if len(reL) >= len(x_ref):
    #     relok = reL[:len(x_ref)]
    # else:
    #     relok = np.append(reL, np.ones(len(x_ref)-len(reL)))

    print('new')
    hrs_new = []
    labs_new = []
    sq_diff = []
    for n_top in allntops:
        alpha_fit = alpha_fit_all[(tip, strat, n_top)]

        alpha_ok = npl.alpha_from_fit(temp, surf_temp, lamb, alpha_fit[cco2], alpha_max = alpha_fit[('max', cco2)], alpha_min = alpha_fit[('min', cco2)], alpha_cose = alpha_fit, alt2 = alt2, n_top = n_top, method = strat)

        hr_calc = npl.hr_reparam_low(cco2, temp, surf_temp, regrcoef = regrcoef, nlte_corr = nlte_corr)
        cr_new = npl.recformula(alpha_ok, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, ovmr = ovmr, n_alts_cs = n_alts_cs)

        if n_top == 65:
            debug_recpar = dict()
            debug_recpar['MM'] = MM
            debug_recpar['L_esc'] = L_esc
            debug_recpar['alpha'] = alpha_ok

        #cr_new = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug_alpha = alpha_ok, n_top = n_top, n_alts_cs = n_alts_cs)

        labs_new.append('{}'.format(n_top))
        hrs_new.append(cr_new)

        sq_diff.append(np.sqrt(np.mean((cr_new[alt2:]-hr_ref[alt2:])**2)))

    # print(type(interp_coeffs))
    # cr_new = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, n_top = n_top)
    hr_calc = npl.hr_reparam_low(cco2, temp, surf_temp, regrcoef = regrcoef, nlte_corr = nlte_corr)
    cr_alphaunif = npl.recformula(alpha_unif[n_top][cco2-1], L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, ovmr = ovmr, n_alts_cs = n_alts_cs)

    hr_calc_full, debug_full = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = True)

    hr_calc_noextP = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old, extrap_co2col = False)

    print('done')

    tit = 'co2: {} - atm: {}'.format(cco2, atm)
    xlab = 'CR (K/day)'
    ylab = 'index'

    # labels = ['nlte_ref', 'old'] + labs_new
    # hrs = [hr_ref, crok] + hrs_new
    # colors = ['violet', 'blue', 'red', 'orange', 'forestgreen', 'brown', 'grey']
    labels = ['nlte_ref', 'fomi', 'aunif'] + labs_new + ['new', 'noextP']
    hrs = [hr_ref, crok, cr_alphaunif] + hrs_new + [hr_calc_full, hr_calc_noextP]
    colors = ['violet', 'blue', 'grey'] + npl.color_set(6, cmap = 'autumn')[:5] + ['black', 'grey']

    xlim = (-1200., 50.)
    xlimdiff = (-40, 40)

    fig, a0, a1 = npl.manuel_plot(np.arange(npl.n_alts_all), hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = xlimdiff, xlim = xlim, ylim = (40, 83), linestyles = ['-', '--', '--', ':', ':', ':', ':', ':', (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5))], colors = colors, orizlines = [40, 50, 65])

    xlim = (-200., 20.)
    xlimdiff = (-5, 5)

    figzoom, _, _ = npl.manuel_plot(np.arange(npl.n_alts_all), hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = xlimdiff, xlim = xlim, ylim = (40, 83), linestyles = ['-', '--', '--', ':', ':', ':', ':', ':', (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5))], colors = colors, orizlines = [40, 50, 65])

    if debug:
        return hr_ref, hr_calc_full, hrs_new[2], debug_full, debug_recpar
    else:
        return fig, a0, a1, figzoom, sq_diff


# usa calc_all_plot per capire le differenze fra la strategia selezionata e quella di new_param_full
# la funz di new_param_full dovrebbe essere identica a v10 nl0 65, ma non è

alldiffs = dict()
for tip in ['v8', '9i', 'v10']:
    for strat in ['nl0', '4e']:
        figs = []
        figszoo = []
        for cco2 in range(1, npl.n_co2prof+1):
            a0s = []
            a1s = []
            diffs = []
            for ii, atm in enumerate(allatms):

                fig, a0, a1, figzoom, sqd = calc_all_plot(cco2, atm, tip, strat, debug = False)

                figs.append(fig)
                figszoo.append(figzoom)
                a0s.append(a0)
                a1s.append(a1)
                diffs.append(sqd)

            alldiffs[(tip, strat, cco2)] = np.array(diffs)

            npl.adjust_ax_scale(a0s)
            npl.adjust_ax_scale(a1s)

        npl.plot_pdfpages(cart_out_F + 'check_fomialpha_refatm_{}_{}_allntop_extP.pdf'.format(tip, strat), figs)
        npl.plot_pdfpages(cart_out_F + 'check_fomialpha_refatm_{}_{}_allntop_extP_zoom.pdf'.format(tip, strat), figszoo)


print('\n All atmospheres: best strategy \n')
filo.write('\n All atmospheres: best strategy \n')

best_for_atm = dict()
cosett = [(tip, strat) for tip in ['v8', '9i', 'v10'] for strat in ['nl0', '4e']]

for ii, atm in enumerate(allatms):
    for cco2 in range(1, npl.n_co2prof+1):
        atmdiffs = np.stack([alldiffs[(tip, strat, cco2)][ii] for tip in ['v8', '9i', 'v10'] for strat in ['nl0', '4e']])
        oknto = np.argmin(atmdiffs, axis = 1)
        okvals = np.min(atmdiffs, axis = 1)

        beststrat = cosett[np.argmin(okvals)]
        bestnto = allntops[oknto[np.argmin(okvals)]]

        min_err = np.min(atmdiffs)

        best_for_atm[(atm, cco2)] = ((beststrat[0], beststrat[1], bestnto), min_err)
        print('{} {}: best {} {} {} -> {:6.3f} K\n'.format(atm, cco2, beststrat[0], beststrat[1], bestnto, min_err))
        filo.write('{} {}: best {} {} {} -> {:6.3f} K\n'.format(atm, cco2, beststrat[0], beststrat[1], bestnto, min_err))

#print(best_for_atm)


print('\n All CO2, mean error of atmospheres: best strategy \n')
filo.write('\n All CO2, mean error of atmospheres: best strategy \n')

best_for_cco2 = dict()
for cco2 in range(1, npl.n_co2prof+1):
    atmdiffs = np.stack([np.mean(alldiffs[(tip, strat, cco2)], axis = 0) for tip in ['v8', '9i', 'v10'] for strat in ['nl0', '4e']])
    oknto = np.argmin(atmdiffs, axis = 1)
    okvals = np.min(atmdiffs, axis = 1)

    beststrat = cosett[np.argmin(okvals)]
    bestnto = allntops[oknto[np.argmin(okvals)]]

    min_err = np.min(atmdiffs)

    best_for_cco2[cco2] = ((beststrat[0], beststrat[1], bestnto), min_err)
    print('{}: best {} {} {} -> {:6.3f} K\n'.format(cco2, beststrat[0], beststrat[1], bestnto, min_err))
    filo.write('{}: best {} {} {} -> {:6.3f} K\n'.format(cco2, beststrat[0], beststrat[1], bestnto, min_err))

#print(best_for_cco2)
co2weights = np.array([0.1, 0.5, 1., 1., 0.5, 0.5, 0.1, 0.1])

atmdiffs_all = np.stack([[np.mean(alldiffs[(tip, strat, cco2)], axis = 0) for tip in ['v8', '9i', 'v10'] for strat in ['nl0', '4e']] for cco2 in range(1, npl.n_co2prof+1)])

atmdiffs = np.average(atmdiffs_all, weights = co2weights, axis = 0)
oknto = np.argmin(atmdiffs, axis = 1)
okvals = np.min(atmdiffs, axis = 1)

beststrat = cosett[np.argmin(okvals)]
bestnto = allntops[oknto[np.argmin(okvals)]]

min_err = np.min(atmdiffs)

print('ALL, CO2 weighted, no atm weight: best {} {} {} -> {:6.3f} K\n'.format(beststrat[0], beststrat[1], bestnto, min_err))
filo.write('ALL, CO2 weighted, no atm weight: best {} {} {} -> {:6.3f} K\n'.format(beststrat[0], beststrat[1], bestnto, min_err))

atmdiffs = np.mean(atmdiffs_all, axis = 0)
oknto = np.argmin(atmdiffs, axis = 1)
okvals = np.min(atmdiffs, axis = 1)

beststrat = cosett[np.argmin(okvals)]
bestnto = allntops[oknto[np.argmin(okvals)]]

min_err = np.min(atmdiffs)

print('ALL, no weight: best {} {} {} -> {:6.3f} K\n'.format(beststrat[0], beststrat[1], bestnto, min_err))
filo.write('ALL, no weight: best {} {} {} -> {:6.3f} K\n'.format(beststrat[0], beststrat[1], bestnto, min_err))


weights = np.array([1, 1, 1, 1, 0.2, 0.2])
print('\n All CO2, weighted error of atmospheres: best strategy \n')
filo.write('\n All CO2, weighted error of atmospheres: best strategy \n')

best_for_cco2_max = dict()
for cco2 in range(1, npl.n_co2prof+1):
    atmdiffs = np.stack([np.average(alldiffs[(tip, strat, cco2)], weights = weights, axis = 0) for tip in ['v8', '9i', 'v10'] for strat in ['nl0', '4e']])
    oknto = np.argmin(atmdiffs, axis = 1)
    okvals = np.min(atmdiffs, axis = 1)

    beststrat = cosett[np.argmin(okvals)]
    bestnto = allntops[oknto[np.argmin(okvals)]]

    min_err = np.min(atmdiffs)

    best_for_cco2_max[cco2] = ((beststrat[0], beststrat[1], bestnto), min_err)
    print('{}: best {} {} {} -> {:6.3f} K\n'.format(cco2, beststrat[0], beststrat[1], bestnto, min_err))
    filo.write('{}: best {} {} {} -> {:6.3f} K\n'.format(cco2, beststrat[0], beststrat[1], bestnto, min_err))


atmdiffs_all = np.stack([[np.average(alldiffs[(tip, strat, cco2)], weights = weights, axis = 0) for tip in ['v8', '9i', 'v10'] for strat in ['nl0', '4e']] for cco2 in range(1, npl.n_co2prof+1)])

atmdiffs = np.average(atmdiffs_all, weights = co2weights, axis = 0)
oknto = np.argmin(atmdiffs, axis = 1)
okvals = np.min(atmdiffs, axis = 1)

beststrat = cosett[np.argmin(okvals)]
bestnto = allntops[oknto[np.argmin(okvals)]]

min_err = np.min(atmdiffs)

print('ALL, atm and CO2 weighted: best {} {} {} -> {:6.3f} K\n'.format(beststrat[0], beststrat[1], bestnto, min_err))
filo.write('ALL, atm and CO2 weighted: best {} {} {} -> {:6.3f} K\n'.format(beststrat[0], beststrat[1], bestnto, min_err))

atmdiffs = np.mean(atmdiffs_all, axis = 0)
oknto = np.argmin(atmdiffs, axis = 1)
okvals = np.min(atmdiffs, axis = 1)

beststrat = cosett[np.argmin(okvals)]
bestnto = allntops[oknto[np.argmin(okvals)]]

min_err = np.min(atmdiffs)

print('ALL, atm weight, no CO2 weight: best {} {} {} -> {:6.3f} K\n'.format(beststrat[0], beststrat[1], bestnto, min_err))
filo.write('ALL, atm weight, no CO2 weight: best {} {} {} -> {:6.3f} K\n'.format(beststrat[0], beststrat[1], bestnto, min_err))

#print(best_for_cco2_max)
sys.exit()


for (tip, strat) in cosett:
    print('\n All CO2, mean/max error of atmospheres with {}, {}, 65 \n'.format(tip, strat))
    filo.write('\n All CO2, mean/max error of atmospheres with {}, {}, 65 \n'.format(tip, strat))

    for cco2 in range(1, npl.n_co2prof+1):
        meadiff = np.mean(alldiffs[(tip, strat, cco2)][:, 2])
        maxdiff = np.max(alldiffs[(tip, strat, cco2)][:, 2])
        atmmax = allatms[np.argmax(alldiffs[(tip, strat, cco2)][:, 2])]
        print('{}: {:6.3f} K.   worst atm: {}, {:6.3f} K\n'.format(cco2, meadiff, atmmax, maxdiff))
        filo.write('{}: {:6.3f} K.   worst atm: {}, {:6.3f} K\n'.format(cco2, meadiff, atmmax, maxdiff))

#print(best_alt_for_atm)
print('\n Best alt for atm with v8, nl0 strategy \n')
filo.write('\n Best alt for atm with v8, nl0 strategy \n')

best_alt_for_atm = dict()
for ii, atm in enumerate(allatms):
    for cco2 in range(1, npl.n_co2prof+1):
        atmdiffs = alldiffs[('v8', 'nl0', cco2)][ii]
        oknto = np.argmin(atmdiffs)
        min_err = np.min(atmdiffs)

        bestnto = allntops[oknto]

        best_alt_for_atm[(atm, cco2)] = (bestnto, min_err)
        print('{} {}: {}. {:6.3f} K'.format(cco2, atm, bestnto, min_err))
        filo.write('{} {}: {}. {:6.3f} K\n'.format(cco2, atm, bestnto, min_err))
    print('\n')

filo.close()
