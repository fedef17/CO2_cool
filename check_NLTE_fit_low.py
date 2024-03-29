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
import newparam_lib as npl

if os.uname()[1] == 'xaru':
    cart_base = '/home/fedef/Research/'
elif os.uname()[1] == 'hobbes':
    cart_base = '/home/fabiano/Research/'
else:
    raise ValueError('Unknown platform {}. Specify paths!'.format(os.uname()[1]))

# sys.path.insert(0, cart_base + 'git/SpectRobot/')
# sys.path.insert(0, cart_base + 'git/pythall/')
cart_out = cart_base + 'lavori/CO2_cooling/new_param/LTE/'
cart_out_2 = cart_base + 'lavori/CO2_cooling/new_param/NLTE/'

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

n_alts = 54

all_alts = atm_pt[('mle', 'alts')]
alts = atm_pt[('mle', 'alts')][:n_alts]

pres = atm_pt[('mle', 'pres')]
x = np.log(1000./pres)
n_alts_trlo = np.sum(x < 12.5)
print('low trans at {}'.format(alts[n_alts_trlo]))

all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

n_alts_lte = 40
max_alts = 55 ### IMPORTANT! max alts for hr_from_ab

figs = []
a0s = []
a1s = []

tot_coeff_co2 = pickle.load(open(cart_out + 'tot_coeffs_co2_v2_LTE.p', 'rb'))
varfit_xis_4_nlte = pickle.load(open(cart_out_2+'varfit_NLTE_v4c.p', 'rb'))
varfit_xis_5_nlte = pickle.load(open(cart_out_2+'varfit_NLTE_v5c.p', 'rb'))

varfit_xis_wiatm = dict()
for iatm in range(6):
    varfit_xis_wiatm[iatm] = pickle.load(open(cart_out_2 + 'varfit_NLTE_iatm{}.p'.format(iatm), 'rb'))

#all_coeffs_nlte_v4 = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE_fitv4.p', 'rb'))
#all_coeffs_nlte_v5 = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE_fitv5.p', 'rb'))

for cco2 in allco2:
    #acoeff, bcoeff, asurf, bsurf = npl.ab_from_xi_abfit_fromdict(varfit_xis_4_nlte, cco2, all_coeffs = all_coeffs_nlte_v4)
    acoeff, bcoeff, asurf, bsurf = npl.ab_from_xi_abfit_fromdict(varfit_xis_4_nlte, cco2, all_coeffs = all_coeffs_nlte)
    acoeff[max_alts:, :] = 0.
    bcoeff[max_alts:, :] = 0.
    acoeff[:, max_alts:] = 0.
    bcoeff[:, max_alts:] = 0.
    asurf[max_alts:] = 0.
    bsurf[max_alts:] = 0.
    tot_coeff_co2[('varfit4_nlte', 'acoeff', cco2)] = acoeff
    tot_coeff_co2[('varfit4_nlte', 'bcoeff', cco2)] = bcoeff
    tot_coeff_co2[('varfit4_nlte', 'asurf', cco2)] = asurf
    tot_coeff_co2[('varfit4_nlte', 'bsurf', cco2)] = bsurf

    #acoeff, bcoeff, asurf, bsurf = npl.ab_from_xi_abfit_fromdict(varfit_xis_5_nlte, cco2, all_coeffs = all_coeffs_nlte_v5)
    acoeff, bcoeff, asurf, bsurf = npl.ab_from_xi_abfit_fromdict(varfit_xis_5_nlte, cco2, all_coeffs = all_coeffs_nlte)
    acoeff[max_alts:, :] = 0.
    bcoeff[max_alts:, :] = 0.
    acoeff[:, max_alts:] = 0.
    bcoeff[:, max_alts:] = 0.
    asurf[max_alts:] = 0.
    bsurf[max_alts:] = 0.
    tot_coeff_co2[('varfit5_nlte', 'acoeff', cco2)] = acoeff
    tot_coeff_co2[('varfit5_nlte', 'bcoeff', cco2)] = bcoeff
    tot_coeff_co2[('varfit5_nlte', 'asurf', cco2)] = asurf
    tot_coeff_co2[('varfit5_nlte', 'bsurf', cco2)] = bsurf

    for iatm in range(6):
        acoeff, bcoeff, asurf, bsurf = npl.ab_from_xi_abfit_fromdict(varfit_xis_wiatm[iatm], cco2, all_coeffs = all_coeffs_nlte)
        tot_coeff_co2[('varfit_wiatm{}_nlte'.format(iatm), 'acoeff', cco2)] = acoeff
        tot_coeff_co2[('varfit_wiatm{}_nlte'.format(iatm), 'bcoeff', cco2)] = bcoeff
        tot_coeff_co2[('varfit_wiatm{}_nlte'.format(iatm), 'asurf', cco2)] = asurf
        tot_coeff_co2[('varfit_wiatm{}_nlte'.format(iatm), 'bsurf', cco2)] = bsurf

    acoeff, bcoeff, asurf, bsurf = npl.ab_from_xi_abfit_fromdict(varfit_xis_5_nlte, cco2, all_coeffs = all_coeffs_nlte, faircoeff = True)
    tot_coeff_co2[('faircoeff_nlte', 'acoeff', cco2)] = acoeff
    tot_coeff_co2[('faircoeff_nlte', 'bcoeff', cco2)] = bcoeff
    tot_coeff_co2[('faircoeff_nlte', 'asurf', cco2)] = asurf
    tot_coeff_co2[('faircoeff_nlte', 'bsurf', cco2)] = bsurf

pickle.dump(tot_coeff_co2, open(cart_out_2 + 'tot_coeffs_co2_NLTE.p', 'wb'))
tot_coeff_co2 = pickle.load(open(cart_out_2 + 'tot_coeffs_co2_NLTE.p', 'rb'))

# poi fai un check please con npl.coeff_from_interp_log()
figs = []
figs2 = []
a0s = []
a1s = []

co2profs = [atm_pt[('mle', cco2, 'co2')] for cco2 in range(1,npl.n_co2prof+1)]

fit_score = dict()
#alltips = ['varfit4', 'varfit5', 'varfit4_nlte', 'varfit5_nlte']
alltips = ['varfit4_nlte', 'varfit5_nlte'] #+ ['varfit_wiatm{}_nlte'.format(iatm) for iatm in range(6)]
for tip in alltips:
    for sco in ['trans', 'lte+trans']:
        for cos in ['std', 'max']:
            fit_score[(tip, sco, cos)] = []

for sco in ['trans', 'lte+trans']:
    for cos in ['std', 'max']:
        fit_score[('fomi', sco, cos)] = []

alt0 = 8

for cco2 in range(1,npl.n_co2prof+1):
    co2pr = co2profs[cco2-1]

    for atm in allatms:
        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]

        tit = 'co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'Alt (km)'

        hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_nlte')][:n_alts]
        hr_lte = all_coeffs[(atm, cco2, 'hr_ref')][:n_alts]

        hr_calcs = []
        for tip in alltips:
            acoeff_cco2 = tot_coeff_co2[(tip, 'acoeff', cco2)]
            bcoeff_cco2 = tot_coeff_co2[(tip, 'bcoeff', cco2)]
            asurf_cco2 = tot_coeff_co2[(tip, 'asurf', cco2)]
            bsurf_cco2 = tot_coeff_co2[(tip, 'bsurf', cco2)]

            hr_calc = npl.hr_from_ab(acoeff_cco2, bcoeff_cco2, asurf_cco2, bsurf_cco2, temp, surf_temp, max_alts = max_alts)[:n_alts]
            hr_calcs.append(hr_calc)
            fit_score[(tip, 'trans', 'std')].append(np.sqrt(np.mean((hr_calc[n_alts_lte:n_alts_trlo]-hr_ref[n_alts_lte:n_alts_trlo])**2)))
            fit_score[(tip, 'lte+trans', 'std')].append(np.sqrt(np.mean((hr_calc[alt0:n_alts_trlo]-hr_ref[alt0:n_alts_trlo])**2)))

            fit_score[(tip, 'trans', 'max')].append(np.max(np.abs(hr_calc[n_alts_lte:n_alts_trlo]-hr_ref[n_alts_lte:n_alts_trlo])))
            fit_score[(tip, 'lte+trans', 'max')].append(np.max(np.abs(hr_calc[alt0:n_alts_trlo]-hr_ref[alt0:n_alts_trlo])))

        pres = atm_pt[(atm, 'pres')]
        print(np.median(co2pr))
        ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
        o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
        n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]

        alt_fomi, x_fomi, hr_fomi = npl.old_param(all_alts, temp, pres, co2pr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr, input_in_ppm = False, cart_run_fomi = cart_base + 'lavori/CO2_cooling/cart_run_fomi/')
        oldco = spline(alt_fomi, hr_fomi)
        hr_fomi = oldco(alts)

        fit_score[('fomi', 'trans', 'std')].append(np.sqrt(np.mean((hr_fomi[n_alts_lte:n_alts_trlo]-hr_ref[n_alts_lte:n_alts_trlo])**2)))
        fit_score[('fomi', 'lte+trans', 'std')].append(np.sqrt(np.mean((hr_fomi[alt0:n_alts_trlo]-hr_ref[alt0:n_alts_trlo])**2)))

        fit_score[('fomi', 'trans', 'max')].append(np.max(np.abs(hr_fomi[n_alts_lte:n_alts_trlo]-hr_ref[n_alts_lte:n_alts_trlo])))
        fit_score[('fomi', 'lte+trans', 'max')].append(np.max(np.abs(hr_fomi[alt0:n_alts_trlo]-hr_ref[alt0:n_alts_trlo])))


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
        hr_ab_rescaled = []
        for pio in ['']:#, '_smoo', '_cut']:
            acoeff_cco2 = all_coeffs_nlte[(atm, cco2, 'acoeff'+pio)]
            bcoeff_cco2 = all_coeffs_nlte[(atm, cco2, 'bcoeff'+pio)]
            asurf_cco2 = all_coeffs_nlte[(atm, cco2, 'asurf'+pio)]
            bsurf_cco2 = all_coeffs_nlte[(atm, cco2, 'bsurf'+pio)]

            hr_calc = npl.hr_from_ab(acoeff_cco2, bcoeff_cco2, asurf_cco2, bsurf_cco2, temp, surf_temp, max_alts = max_alts)[:n_alts]
            hr_ab_rescaled.append(hr_calc)

        ksk = np.mean(np.abs(hr_ref-hr_ab_rescaled[0]))
        if ksk < 0.01:
            print(cco2, atm, 'OK!')
        else:
            print('---------> ', cco2, atm, 'NOUUUUUUU [{:.0f}]'.format(ksk), np.mean(hr_ab_rescaled[0]), np.mean(hr_ref))
            if np.sum(np.abs(hr_ref+hr_ab_rescaled[0])) < 0.01:
                print('-----------> ok!! this is crazyyyy')


        #labels = ['nlte_ref', 'lte_ref'] + alltips + ['fomi rescale (no fit)', 'old param']
        #hrs = [hr_ref, hr_lte] + hr_calcs + hr_ab_rescaled + [hr_fomi]
        labels = ['nlte_ref'] + alltips + ['lte_ref', 'old param']
        hrs = [hr_ref] + hr_calcs + [hr_lte, hr_fomi]
        #labels = ['ref'] + alltips + ['fomi rescale (no fit)', 'old param']
        fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-3, 3), xlim = (-40, 10), ylim = (40, 90), linestyles = ['-', '-', '-', ':', ':'])

        figs2.append(fig)
        a0s.append(a0)
        a1s.append(a1)

npl.adjust_ax_scale(a0s)
npl.adjust_ax_scale(a1s)
# npl.plot_pdfpages(cart_out + 'check_newparam_LTE_final_newvsold.pdf', figs)

npl.plot_pdfpages(cart_out_2 + 'check_newparam_NLTE_lowtrans.pdf', figs2)

alltips.append('fomi')

for cos in ['std', 'max']:
    if cos == 'std':
        print('Average stddev of param in region.\n')
    else:
        print('Max absolute error of param in region.\n')

    for sco in ['trans', 'lte+trans']:
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


print('\n\n -------------------------------- \n\n')
alltips = ['varfit4_nlte', 'varfit5_nlte', 'fomi']
vsho = ['v4', 'v5', 'old']

for cos in ['std', 'max']:
    if cos == 'std':
        print('\n\nAverage stddev of param in region.\n')
    else:
        print('\n\nMax absolute error of param in region.\n')

    for sco in ['trans', 'lte+trans']:
        print('---------------- \n')
        print(sco + ' region \n')

        iooo = 0
        for cco2 in range(1, npl.n_co2prof+1):
            print('\n')
            for atm in allatms:
                lui1 = fit_score[(alltips[0], sco, cos)][iooo]
                lui2 = fit_score[(alltips[1], sco, cos)][iooo]
                lui3 = fit_score[(alltips[2], sco, cos)][iooo]
                print('{}, {}. v4: {:6.3f}, v5: {:6.3f}, old: {:6.3f} K  -----------> mejor? {}'.format(cco2, atm, lui1, lui2, lui3, vsho[np.argmin([lui1, lui2, lui3])]))
                iooo += 1
