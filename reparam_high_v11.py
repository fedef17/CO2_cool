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
for n_top in [65, 60, 63, 67, 70]:
    print('------------------- \n {} \n ---------------------'.format(n_top))
    ########### Qui c'è la parte del fit dell'alpha
    # alpha FIT!

    n_trans = n_top-alt2+1

    #bounds = (0.1*np.ones(n_trans), 100*np.ones(n_trans))
    bounds = (0.1*np.ones(n_trans), 100*np.ones(n_trans))

    for afit, atmw in zip(['a{}'.format(i) for i in range(5)], [np.ones(6), [0.3, 0.1, 0.1, 0.4, 0.05, 0.05], np.array([0.3, 0.1, 0.1, 0.4, 0.05, 0.05])**2, [0., 0.1, 0.1, 0., 1, 1], [1, 0., 0., 1, 0., 0.]]):
        atmweights = atmw

        alpha_unif = []
        alpha_dic_atm = dict()
        start = np.ones(n_trans)
        name_escape_fun = 'L_esc_all_extP'
        for cco2 in range(1, npl.n_co2prof+1):
            result = least_squares(npl.delta_alpha_rec2_recf, start, args=(cco2, cose_upper_atm, alt2, n_top, atmweights, all_coeffs_nlte, atm_pt, name_escape_fun, ), verbose=1, method = 'trf', bounds = bounds, max_nfev = 20000, ftol = 1.e-10, gtol = 1.e-10, xtol = 1.e-10)
            alpha_unif.append(result.x)

            if do_single:
                alphas = []
                for atm in allatms:
                    print(atm, cco2)
                    cose_upper_atm[(atm, cco2, 'eps125')] = all_coeffs_nlte[(atm, cco2, 'hr_ref')][alt2-1] # Trying with the reference HR
                    ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')][alt2-1:n_top]
                    result = least_squares(npl.delta_alpha_rec2_atm, start, args=(atm, cco2, cose_upper_atm, alt2, n_top, atmweights, all_coeffs_nlte, atm_pt, name_escape_fun, ovmr, ), verbose=1, method = 'trf', bounds = bounds, max_nfev = 20000)#, gtol = gtol, xtol = xtol)
                    #result = least_squares(npl.delta_alpha_rec2, 10*np.ones(n_trans), args=(cco2, cose_upper_atm, n_alts_trlo, n_alts_trhi, atmweights, all_coeffs_nlte, atm_pt, ), verbose=1, method = 'lm')
                    print('least_squares', result)
                    alphas.append(result.x)

                alpha_dic_atm[cco2] = np.stack(alphas)

        alpha_unif = np.stack(alpha_unif)
        pickle.dump(alpha_unif, open(cart_out_rep + 'alpha_unif_v{}_top{}.p'.format(afit[-1], n_top), 'wb'))
        alpha_dic[(afit, n_top)] = alpha_unif

    continue

    pickle.dump([alpha_unif, alpha_dic_atm], open(cart_out_rep + 'alpha_singleatm_v2_top{}.p'.format(n_top), 'wb'))

    alpha_unif, alpha_dic_atm = pickle.load(open(cart_out_rep + 'alpha_singleatm_v2_top{}.p'.format(n_top), 'rb'))

    #### OK, and now.... regression model! with pc0, pc1 e tempgrad
    tempgrad = np.stack([np.gradient(atm_pt[(atm, 'temp')])[alt2:n_top+1] for atm in allatms])
    temps = np.stack([atm_pt[(atm, 'temp')][alt2:n_top+1] for atm in allatms])


    kbc = 0.69503
    E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition
    phifunz = np.exp(-E_fun/(kbc*temps))

    allcorrs = dict()
    eofs_all = dict()

    moddict = dict()

    popup = np.array([phifunz[ii]*cose_upper_atm[(atm, cco2, 'lamb')][alt2:n_top+1]/1.5988 for ii, atm in enumerate(allatms) for cco2 in range(1,npl.n_co2prof+1)])

    popup_mean = np.mean(popup, axis = 0)
    popup_anom = popup-popup_mean

    solver_pop = Eof(popup_anom)

    fig = plt.figure()
    for i in range(4):
        plt.plot(solver_pop.eofs()[i], np.arange(alt2, n_top+1), label = str(i))
    plt.title('popup eofs')
    fig.savefig(cart_out_F + 'popup_eofs_allco2_top{}.pdf'.format(n_top))

    figs3 = []
    alpha_fit = dict()
    alpha_fit_nl0 = dict()
    alpha_fit2 = dict()
    for cco2 in range(1,npl.n_co2prof+1):
        alphaok = alpha_dic_atm[cco2]
        alpha_min = np.min(alphaok, axis = 0)
        alpha_max = np.max(alphaok, axis = 0)

        lambdivA = np.array([cose_upper_atm[(atm, cco2, 'lamb')][alt2:n_top+1]/1.5988 for atm in allatms])
        popup = lambdivA*phifunz

        Xprods = []
        for ee in range(4):
            dotprods = np.array([np.dot(pop-popup_mean, solver_pop.eofs(eofscaling=1)[ee]) for pop in popup])
            Xprods.append(dotprods)

        X = np.stack(Xprods).T
        #X = np.stack([pop_x0, pop_x1, pop_x2, pop_x3]).T

        weights = np.array([1, 1, 1, 1, 1, 1])

        ### model 1: multi-linear regression of first 4 popup pcs
        scores = []
        ints = []
        coefs = []
        for ii in range(n_trans):
            Y = np.stack(alphaok[:,ii])
            model1 = LinearRegression().fit(X, Y, weights)
            print(ii, model1.score(X, Y))
            scores.append(model1.score(X, Y))
            ints.append(model1.intercept_)
            coefs.append(model1.coef_)
            print('\n')

        alpha_fit[cco2] = np.concatenate([np.array(ints)[:, np.newaxis], np.stack(coefs)], axis = 1)

        ### model 2: parabolic fit to 1st pc, multi-linear regression of residuals with other 2 popup pcs
        Xprods = []
        for ee in range(2):#3):
            dotprods = np.array([np.dot(pop-popup_mean, solver_pop.eofs(eofscaling=1)[ee]) for pop in popup])
            Xprods.append(dotprods)
            #if ee == 0:
            Xprods.append(dotprods**2)

        X2 = np.stack(Xprods).T

        scores = []
        ints = []
        coefs = []
        for ii in range(n_trans):
            Y = np.stack(alphaok[:,ii])
            model1 = LinearRegression().fit(X2, Y, weights)
            print(ii, model1.score(X2, Y))
            scores.append(model1.score(X2, Y))
            ints.append(model1.intercept_)
            coefs.append(model1.coef_)
            print('\n')

        alpha_fit2[cco2] = np.concatenate([np.array(ints)[:, np.newaxis], np.stack(coefs)], axis = 1)

        fig3, ax3 = plt.subplots()
        for ii, (atm, col) in enumerate(zip(allatms, npl.color_set(6))):
            ls = '--'
            alpha = alpha_fit[cco2][:, 0] + np.sum(alpha_fit[cco2][:, 1:] * X[ii][np.newaxis, :], axis = 1)

            alpha[alpha < alpha_min] = alpha_min[alpha < alpha_min]
            alpha[alpha > alpha_max] = alpha_max[alpha > alpha_max]

            ax3.plot((alpha-alphaok[ii, :])/alphaok[ii, :], np.arange(alt2, n_top+1), color = col, linestyle = ls, label = atm)

            # ax3.plot((alpha_unif[cco2-1]-alphaok[ii, :])/alphaok[ii, :], np.arange(alt2, n_top+1), color = col, linestyle = ':')

        # plt.legend()
        # plt.title('cco2: ' + str(cco2) + '- mod lin')
        # plt.xlabel('relative error in alpha')
        # figs3.append(fig3)

        #fig3, ax3 = plt.subplots()
        for ii, (atm, col) in enumerate(zip(allatms, npl.color_set(6))):
            ls = ':'
            alpha = alpha_fit2[cco2][:, 0] + np.sum(alpha_fit2[cco2][:, 1:] * X2[ii][np.newaxis, :], axis = 1)
            alpha[alpha < alpha_min] = alpha_min[alpha < alpha_min]
            alpha[alpha > alpha_max] = alpha_max[alpha > alpha_max]
            #alpha = np.array([np.polyval(alpha_fit_nl0[cco2][izz, :], X[ii, 0]) for izz in range(len(alpha_fit_nl0[cco2]))]) + alpha_fit2[cco2][:, 0] +  np.sum(alpha_fit2[cco2][:, 1:] * X[ii][np.newaxis, 1:-1], axis = 1)
            ax3.plot((alpha-alphaok[ii, :])/alphaok[ii, :], np.arange(alt2, n_top+1), color = col, linestyle = ls, label = atm)

            # ax3.plot((alpha_unif[cco2-1]-alphaok[ii, :])/alphaok[ii, :], np.arange(alt2, n_top+1), color = col, linestyle = ':')

        plt.legend()
        plt.title('cco2: ' + str(cco2) + '- lin vs non lin')
        plt.xlabel('relative error in alpha')
        figs3.append(fig3)


    npl.plot_pdfpages(cart_out_rep + 'check_alpha_popup_relerr_v8_top{}.pdf'.format(n_top), figs3)

    #sys.exit()
    #####################################################################

    ####################################################################

    alpha_fit['popup_mean'] = popup_mean
    alpha_fit['eof0'] = solver_pop.eofs(eofscaling=1)[0]
    alpha_fit['eof1'] = solver_pop.eofs(eofscaling=1)[1]
    alpha_fit['eof2'] = solver_pop.eofs(eofscaling=1)[2]
    alpha_fit['eof3'] = solver_pop.eofs(eofscaling=1)[3]

    for cco2 in range(1, npl.n_co2prof+1):
        alpha_fit[('min', cco2)] = np.min(alpha_dic_atm[cco2], axis = 0)
        alpha_fit[('max', cco2)] = np.max(alpha_dic_atm[cco2], axis = 0)

    pickle.dump(alpha_fit, open(cart_out_rep + 'alpha_fit_4e_v8_top{}.p'.format(n_top), 'wb'))

    alpha_fit2['popup_mean'] = popup_mean
    alpha_fit2['eof0'] = solver_pop.eofs(eofscaling=1)[0]
    alpha_fit2['eof1'] = solver_pop.eofs(eofscaling=1)[1]

    for cco2 in range(1, npl.n_co2prof+1):
        alpha_fit2[('min', cco2)] = np.min(alpha_dic_atm[cco2], axis = 0)
        alpha_fit2[('max', cco2)] = np.max(alpha_dic_atm[cco2], axis = 0)
    pickle.dump(alpha_fit2, open(cart_out_rep + 'alpha_fit_nl0_v8_top{}.p'.format(n_top), 'wb'))


    ############################################################
    tot_coeff_co2 = pickle.load(open(cart_out_2 + 'tot_coeffs_co2_NLTE.p', 'rb'))

    figs = []
    a0s = []
    a1s = []
    for cco2 in range(1, npl.n_co2prof+1):
        alpha_min = np.min(alpha_dic_atm[cco2], axis = 0)
        alpha_max = np.max(alpha_dic_atm[cco2], axis = 0)

        for ii, atm in enumerate(allatms):
            temp = atm_pt[(atm, 'temp')]
            surf_temp = atm_pt[(atm, 'surf_temp')]
            pres = atm_pt[(atm, 'pres')]

            co2vmr = atm_pt[(atm, cco2, 'co2')]
            ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
            o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
            n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]

            L_esc = cose_upper_atm[(atm, cco2, 'L_esc_all_extP')]
            lamb = cose_upper_atm[(atm, cco2, 'lamb')]
            MM = cose_upper_atm[(atm, cco2, 'MM')]

            hr_calc = npl.hr_reparam_low(cco2, temp, surf_temp, regrcoef = regrcoef, nlte_corr = nlte_corr)

            # hra, hrb = npl.hr_from_ab_diagnondiag(all_coeffs[(atm, cco2, 'acoeff')], all_coeffs[(atm, cco2, 'bcoeff')], all_coeffs[(atm, cco2, 'asurf')], all_coeffs[(atm, cco2, 'bsurf')], atm_pt[(atm, 'temp')], atm_pt[(atm, 'surf_temp')], max_alts = npl.n_alts_all)
            # hra = hra[alt2:n_top+1]
            # hrb = hrb[alt2:n_top+1]

            # population upper state
            phifunz = np.exp(-E_fun/(kbc*temp[alt2:n_top+1]))
            lambdivA = lamb[alt2:n_top+1]/1.5988
            popup = lambdivA*phifunz

            dotprods = np.array([np.dot(popup-popup_mean, eoff) for eoff in solver_pop.eofs(eofscaling=1)[:4]])

            alpha6 = alpha_fit[cco2][:, 0] + np.sum(alpha_fit[cco2][:, 1:] * dotprods[np.newaxis, :], axis = 1)

            print('setting constraint on alpha! check this part')
            alpha6[alpha6 < alpha_min] = alpha_min[alpha6 < alpha_min]
            alpha6[alpha6 > alpha_max] = alpha_max[alpha6 > alpha_max]

            if cco2 == 7:
                print('AAAAA')
                print(alpha6)

            alphaok = alpha_dic_atm[cco2][ii, :]
            if cco2 == 7: print(alphaok)

            #dotprods2 = np.array([dotprods[0], dotprods[0]**2] + list(dotprods[1:-1]))
            dotprods2 = np.array([dotprods[0], dotprods[0]**2] + [dotprods[1], dotprods[1]**2])
            alpha_nl = alpha_fit2[cco2][:, 0] + np.sum(alpha_fit2[cco2][:, 1:] * dotprods2[np.newaxis, :], axis = 1)

            print('setting constraint on alpha! check this part')
            alpha_nl[alpha_nl < alpha_min] = alpha_min[alpha_nl < alpha_min]
            alpha_nl[alpha_nl > alpha_max] = alpha_max[alpha_nl > alpha_max]
            # alpha_nl = np.array([np.polyval(alpha_fit_nl0[cco2][izz, :], dotprods[0]) for izz in range(len(alpha_fit_nl0[cco2]))]) + alpha_fit2[cco2][:, 0] +  np.sum(alpha_fit2[cco2][:, 1:] * dotprods[np.newaxis, 1:-1], axis = 1)

            #n_top = alt2 + 10
            hr_calc6 = npl.recformula(alpha6, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, ovmr = ovmr, n_alts_cs = n_alts_cs)
            hr_calc_nl = npl.recformula(alpha_nl, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, ovmr = ovmr, n_alts_cs = n_alts_cs)

            # hr_calc_aok = npl.recformula(alphaok, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top)
            hr_calc_aunif = npl.recformula(alpha_unif[cco2-1, :], L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, ovmr = ovmr, n_alts_cs = n_alts_cs)

            hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_ref')]
            #hr_calc_vf5 = npl.new_param_full_old(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr)#, coeffs = coeffs_NLTE)

            # alt_fomi, hr_fomi = npl.old_param(all_alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr)
            # oldco = spline(alt_fomi, hr_fomi)
            # hr_fomi = oldco(alts)

            tit = 'co2: {} - atm: {}'.format(cco2, atm)
            xlab = 'CR (K/day)'
            ylab = 'index'
            # labels = ['nlte_ref'] + [modnams[4]] + [modnams[5]] + ['alphaok', 'vf5', 'auni']
            # hrs = [hr_ref, hr_calc5, hr_calc6, hr_calc_aok, hr_calc_vf5, hr_calc_aunif]
            labels = ['nlte_ref', 'pop_4e_wt', 'pop_nl0_wt', 'aunif']#, 'vf5']#, 'fomi']
            hrs = [hr_ref, hr_calc6, hr_calc_nl, hr_calc_aunif]#, hr_calc_vf5]#, hr_fomi]

            colors = npl.color_set(5)
            fig, a0, a1 = npl.manuel_plot(np.arange(npl.n_alts_all), hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-25, 25), xlim = (-1000, 10), ylim = (40, 80), linestyles = ['-', '--', ':', ':', ':', ':', ':'], colors = colors, orizlines = [40, alt2, n_top], linewidth = 2.)

            figs.append(fig)
            a0s.append(a0)
            a1s.append(a1)

            npl.adjust_ax_scale(a0s)
            npl.adjust_ax_scale(a1s)

    npl.plot_pdfpages(cart_out_F + 'check_reparam_high_v8_top{}.pdf'.format(n_top), figs)
