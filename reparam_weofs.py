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
n_alts = 51
for n_alts in [41, 46, 51, 56, 61, 66]:
    print(n_alts)
    alts = atm_pt[('mle', 'alts')][:n_alts]

    cartou = cart_out_rep + 'alts{}/'.format(n_alts)
    if not os.path.exists(cartou): os.mkdir(cartou)

    temps = [atm_pt[(atm, 'temp')][:n_alts] for atm in allatms]
    temps = np.stack(temps)

    temps_anom = np.stack([atm_pt[(atm, 'temp')][:n_alts]-np.mean(atm_pt[(atm, 'temp')][:n_alts]) for atm in allatms])
    atm_anom_mean = np.mean(temps_anom, axis = 0)

    solver = Eof(temps)
    solver_anom = Eof(temps_anom)

    fig = plt.figure()
    for i, eo in enumerate(solver.eofs()):
        plt.plot(eo, alts, label = i)
    plt.legend()
    fig.savefig(cartou + 'eofs_temps.pdf')

    fig = plt.figure()
    for i, eo in enumerate(solver_anom.eofs()):
        plt.plot(eo, alts, label = i)
    plt.legend()
    fig.savefig(cartou + 'eofs_temps_anom.pdf')

    fig = plt.figure()
    plt.bar(np.arange(6), solver.eigenvalues())
    fig.savefig(cartou + 'eigenvalues_temps.pdf')

    fig = plt.figure()
    plt.bar(np.arange(6), solver_anom.eigenvalues())
    fig.savefig(cartou + 'eigenvalues_temps_anom.pdf')

    fig = plt.figure()
    plt.bar(np.arange(6), solver.varianceFraction())
    fig.savefig(cartou + 'varfrac_temps.pdf')

    fig = plt.figure()
    plt.bar(np.arange(6), solver_anom.varianceFraction())
    fig.savefig(cartou + 'varfrac_temps_anom.pdf')


    fig = plt.figure()
    atm_mean = np.mean(temps, axis = 0)
    for i, pc in enumerate(solver.pcs()[:,0]):
        plt.plot(atm_mean+pc*solver.eofs()[0]-temps[i, :], alts)
    fig.savefig(cartou + 'residual_temps_firstpc.pdf')

    fig = plt.figure()
    atm_mean = np.mean(temps, axis = 0)
    for i, pc in enumerate(solver_anom.pcs()[:,0]):
        plt.plot(atm_anom_mean+pc*solver_anom.eofs()[0]-temps_anom[i, :], alts)
    fig.savefig(cartou + 'residual_temps_anom_firstpc.pdf')

    # plt.figure()
    # for i, pc in enumerate(solver.pcs()[:,:2]):
    #     plt.plot(atm_mean+pc[0]*solver.eofs()[0]+pc[1]*solver.eofs()[1]-temps[i,:], alts)


    # ok so, if keeping only first and second eof I'm able to explain quite a fraction of the variability
    # the coeffs will be written as: C = C0 + alpha*C1 + beta*C2, with C1 and C2 being the pcs of the actual temp profile with respect to the first two eofs. Calculation of C1 and C2 implies two dot products over 66 altitudes. Plus the sum to determine C. Affordable? yes!

    # Now for the coeffs. Are the coeffs pcs linked to the temp pcs? (correlation?). If so, the method could work well!
    cco2 = 7

    surftemps = np.array([atm_pt[(atm, 'surf_temp')] for atm in allatms])

    coefsolv = dict()
    for conam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        acos = np.stack([all_coeffs_nlte[(atm, cco2, conam)] for atm in allatms])
        if acos.ndim == 3:
            acos = acos[:, :n_alts, ...][..., :n_alts]
        else:
            acos = acos[:, :n_alts]

        aco_solver = Eof(acos)
        coefsolv[conam+'_mean'] = np.mean(acos, axis = 0)

        fig = plt.figure()
        plt.bar(np.arange(6), aco_solver.varianceFraction())
        fig.savefig(cartou + 'varfrac_{}.pdf'.format(conam))

        fig = plt.figure()
        plt.plot(aco_solver.eofs()[0], alts)
        fig.savefig(cartou + 'varfrac_{}.pdf'.format(conam))

        if 'coeff' in conam:
            cor0 = np.corrcoef(solver.pcs()[:, 0], aco_solver.pcs()[:, 0])[1,0]
            cor1 = np.corrcoef(solver.pcs()[:, 1], aco_solver.pcs()[:, 1])[1,0]
            cor0_anom = np.corrcoef(solver_anom.pcs()[:, 0], aco_solver.pcs()[:, 0])[1,0]
            cor1_anom = np.corrcoef(solver_anom.pcs()[:, 1], aco_solver.pcs()[:, 1])[1,0]
            print(conam, cor0, cor1, cor0_anom, cor1_anom)
        else:
            cor0 = np.corrcoef(surftemps, aco_solver.pcs()[:, 0])[1,0]
            cor1 = np.corrcoef(surftemps, aco_solver.pcs()[:, 1])[1,0]
            print(conam, cor0, cor1)
        coefsolv[conam] = aco_solver


    x0 = solver_anom.pcs()[:, 0]
    # SIMPLER. Linear (or nonlinear?) regression of coeff with first pc of temp profile
    regrcoef = dict()
    for conam in ['acoeff', 'bcoeff']:
        acos = np.stack([all_coeffs_nlte[(atm, cco2, conam)] for atm in allatms])
        if acos.ndim == 3:
            acos = acos[:, :n_alts, ...][..., :n_alts]
        else:
            acos = acos[:, :n_alts]

        cico, regrco, _, _ = npl.linearregre_coeff(x0, acos)

        corrco = np.empty_like(acos[0])
        for i in range(6):
            for j in range(6):
                corrco[i,j] = np.corrcoef(x0, acos[:, i, j])[1,0]

        regrcoef[(conam, 'R')] = corrco
        regrcoef[(conam, 'c')] = cico
        regrcoef[(conam, 'm')] = regrco


    for conam in ['acoeff', 'bcoeff']:
        fig = plt.figure()
        for ialt, col in zip(range(n_alts), npl.color_set(n_alts)):
            plt.plot(coefsolv[conam].eofs()[0][:, ialt], alts, color = col)
        plt.xlim(-0.02, 0.02)
        plt.title(conam + 'eof 0')
        fig.savefig(cartou + '{}_eof0.pdf'.format(conam))


    for conam in ['acoeff', 'bcoeff']:
        fig = plt.figure()
        for ialt, col in zip(range(n_alts), npl.color_set(n_alts)):
            plt.plot(regrcoef[(conam, 'R')][:, ialt], alts, color = col)
        #plt.xlim(-0.02, 0.02)
        plt.title(conam + 'rcorr')
        fig.savefig(cartou + '{}_rcorr.pdf'.format(conam))

    fig = plt.figure()
    plt.plot(coefsolv['asurf'].eofs()[0], alts, label = 'asurf')
    plt.plot(coefsolv['bsurf'].eofs()[0], alts, label = 'bsurf')
    plt.xscale('log')
    plt.title('asurf and bsurf eof0')
    fig.savefig(cartou + 'surfco_eof0.pdf')

    # ('acoeff', 0.9201600549720309, 0.5650724813187429)
    # ('bcoeff', 0.8852668113273987, 0.40514400023917907)
    # ('asurf', -0.9916467503397747, 0.12197028746306282)
    # ('bsurf', -0.9864472297843829, 0.14140499211950414)

    # the scalar products between the temp anomalies and the first eof of the temperature profile
    dotprods = np.array([np.dot(te-atm_anom_mean, solver_anom.eofs(eofscaling=1)[0]) for te in temps_anom])

    colors = npl.color_set(6)

    linfits = dict()

    for conam in ['acoeff', 'bcoeff']:
        print(conam)
        print(np.corrcoef(dotprods, coefsolv[conam].pcs(pcscaling=1)[:, 0]))

        fig = plt.figure()
        for dotp, pcc, atm, col in zip(dotprods, coefsolv[conam].pcs(pcscaling=1)[:, 0], allatms, colors):
            plt.scatter([dotp], [pcc], color = col, label = atm)
        m, c, err_m, err_c = npl.linear_regre_witherr(dotprods, coefsolv[conam].pcs(pcscaling=1)[:, 0])
        linfits[conam] = (c, m)
        xcoso = np.linspace(-2, 2, 10)
        plt.plot(xcoso, m*xcoso + c)
        print(c, m)
        plt.xlabel('projection of temp. anom on first eof')
        plt.ylabel('first pc of {}'.format(conam))
        plt.legend()
        #npl.custom_legend(fig, colors, allatms, ncol = 3)
        fig.savefig(cartou + 'fit_{}_vs_tempeof.pdf'.format(conam))


    surfanom = surftemps-np.mean(surftemps)
    for conam in ['asurf', 'bsurf']:
        print(conam)
        print(np.corrcoef(surfanom, coefsolv[conam].pcs(pcscaling=1)[:, 0]))

        fig = plt.figure()
        for dotp, pcc, atm, col in zip(surfanom, coefsolv[conam].pcs(pcscaling=1)[:, 0], allatms, colors):
            plt.scatter([dotp], [pcc], color = col, label = atm)
        m, c, err_m, err_c = npl.linear_regre_witherr(surfanom, coefsolv[conam].pcs(pcscaling=1)[:, 0])
        xcoso = np.linspace(np.min(surfanom), np.max(surfanom), 10)
        plt.plot(xcoso, m*xcoso + c)
        print(c, m)
        linfits[conam] = (c, m)
        plt.xlabel('surf. temp. anomaly')
        plt.ylabel('first pc of {}'.format(conam))
        plt.legend()
        #npl.custom_legend(fig, colors, allatms, ncol = 3)
        fig.savefig(cartou + 'fit_{}_vs_surftemp.pdf'.format(conam))

    # for each coeff: mean and first eof
    coef_cose = dict()
    for conam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        coef_cose[conam] = (coefsolv[conam+'_mean'], coefsolv[conam].eofs(eofscaling=1)[0])

    coef_cose['temp'] = (atm_anom_mean, solver_anom.eofs(eofscaling=1)[0]) # mean and first eof
    pickle.dump([coef_cose, linfits], open(cartou + 'reparam_eofcoeffs.p', 'wb'))

    figs = []
    for atm in allatms:
        fig = plt.figure()
        coeff = all_coeffs_nlte[(atm, cco2, 'acoeff')]
        for ialt, col in zip(range(n_alts), npl.color_set(n_alts)):
            plt.plot(coeff[:n_alts, ialt], alts, color = col)
            if ialt > 1 and ialt < n_alts-1:
                if np.abs(coeff[ialt, ialt])/np.abs(np.mean([coeff[ialt-1, ialt-1], coeff[ialt+1, ialt+1]])) > 1.5:
                    print('Atm {}. Unstable ialt {}'.format(atm, ialt))
                    plt.plot(np.mean([coeff[:n_alts, ialt-1], coeff[:n_alts, ialt+1]], axis = 0), alts, color = col, linestyle = '--')
            plt.title('acoeff - ' + atm)
        figs.append(fig)
    npl.plot_pdfpages(cartou + 'acoeff_atmvar.pdf', figs)

    figs = []
    for atm in allatms:
        fig = plt.figure()
        coeff = all_coeffs_nlte[(atm, cco2, 'bcoeff')]
        for ialt, col in zip(range(n_alts), npl.color_set(n_alts)):
            plt.plot(coeff[:n_alts, ialt], alts, color = col)
            if ialt > 1 and ialt < n_alts-1:
                if np.abs(coeff[ialt, ialt])/np.abs(np.mean([coeff[ialt-1, ialt-1], coeff[ialt+1, ialt+1]])) > 1.5:
                    print('Atm {}. Unstable ialt {}'.format(atm, ialt))
                    plt.plot(np.mean([coeff[:n_alts, ialt-1], coeff[:n_alts, ialt+1]], axis = 0), alts, color = col, linestyle = '--')
            plt.title('bcoeff - ' + atm)
        figs.append(fig)
    npl.plot_pdfpages(cartou + 'bcoeff_atmvar.pdf', figs)


# c, m for all coeffs
# acoeff
# (-1.4163957869901234e-17, 0.9208461477620374)
# bcoeff
# (-1.4163957869901234e-17, 0.9208461477620374)
# asurf
# (-6.232141462756543e-16, -0.06910854929014554)
# bsurf
# (-3.1727265628578765e-16, -0.0687461910991221)

#c ~ 0
