#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pickle
from matplotlib import pyplot as plt

from scipy.interpolate import PchipInterpolator as spline
import numpy as np
import os
from scipy import io
from eofs.standard import Eof

import newparam_lib as npl

########################################################
if os.uname()[1] == 'xaru':
    sys.path.insert(0, '/home/fedef/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fedef/Research/git/pythall/')
    cart_base = '/home/fedef/Research/'
elif os.uname()[1] == 'hobbes':
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_base = '/home/fabiano/Research/'

import spect_base_module as sbm

cart_in = cart_base + 'lavori/CO2_cooling/MIPAS_2009/'

savT = io.readsav(cart_in+'CR20090215/L2_20090215_T_521.6', verbose=True)
savCR = io.readsav(cart_in+'CR20090215/L2_20090215_CR-CO2-IR_521.6', verbose=True)
savO = io.readsav(cart_in+'CR20090215/L2_20090215_O_521.6', verbose=True)
savCO2 = io.readsav(cart_in+'CR20090215/L2_20090215_CO2_521.6', verbose=True)

T = savT.result
CR = savCR.result
O = savO.result
CO2 = savCO2.result

############################

cart_np = cart_in + '../new_param/'
cart_out = cart_np + 'LTE/'
cart_out_2 = cart_np + 'NLTE/'
cart_out_rep = cart_np + 'NLTE_reparam/'
cart_out_3 = cart_np + 'NLTE_upper/'

cart_out_F = cart_np + 'newpar_allatm_2/'

allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
#atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweights = np.ones(6)/6.
atmweights = dict(zip(allatms, atmweights))
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

cose_upper_atm = pickle.load(open(cart_out_3 + 'cose_upper_atm.p', 'rb'))

alt2 = 51
n_top = 65
############################

cart_out = cart_base + 'lavori/CO2_cooling/new_param/mipas_check/'

version = '_xinterp_v3'

fil = 'ssw2009_v3_okTOCO2_1e13_newparam{}.p'.format(version)

gigi = pickle.load(open(cart_out+fil, 'rb'))
cose = gigi.dtype.names


obs, old_param, new_param, new_param_fa  = pickle.load(open(cart_out+'out_ssw2009{}.p'.format(version),'rb'))


alts = gigi.altitude[0]

mippres = np.stack(gigi.pressure)
mipx = np.log(1000./mippres)

alt_manuel, mol_vmrs, molist, molnums = sbm.read_input_vmr_man(cart_in + 'gases_120.dat', version = 2)
o2vmr = mol_vmrs['O2']*1.e-6
n2vmr = mol_vmrs['N2']*1.e-6
spl = spline(alt_manuel, o2vmr)
o2vmr = spl(alts)
spl = spline(alt_manuel, n2vmr)
n2vmr = spl(alts)

# zup = np.loadtxt('coso')
# alt_fomi = zup[:, 0]

crun = '/home/fabiano/Research/lavori/CO2_cooling/cart_run_fomi/'
alt_fomi, x_fomi, cr_fomi = npl.old_param(gigi.altitude[0], gigi.temperature[0], gigi.pressure[0], 1.e-6*CO2.target[0], Oprof = 1.e-6*O.target[0], O2prof = o2vmr, N2prof = n2vmr, input_in_ppm = False, cart_run_fomi = crun)

######################

test_recfor = True
nmax = 867

if test_recfor:
    crfom_ok = []
    for crfom in gigi.cr_fomi[:nmax]:
        spl = spline(alt_fomi, crfom)
        crfo2 = spl(alts)
        crfom_ok.append(crfo2)
    crfom_ok = np.stack(crfom_ok)

    ## mipas eof

    crfom_recalc = []
    crfom_old = []
    crfom_new = []
    interp_coeffs = npl.precalc_interp()

    co2cols = []
    co2cols_fomi = []

    xs_ref = []
    #for ii in range(len(gigi.pressure)):
    for ii in range(nmax):
        print(ii)
        if ii % 25 == 0: print(ii)
        ## interpolate to reference grid
        x = mipx[ii]
        temp = gigi.temperature[ii]
        pres = gigi.pressure[ii]
        ovmr = 1.e-6*O.target[ii]
        co2vmr = 1.e-6*CO2.target[ii]

        sys.exit()

        # x_ref_max = 16.735
        x_ref = np.arange(0.125, np.max(x) + 0.001, 0.25)

        spl = spline(x, temp)
        temp_rg = spl(x_ref)

        spl = spline(x, np.log(pres))
        pres_rg = spl(x_ref)
        pres_rg = np.exp(pres_rg)

        spl = spline(x, ovmr)
        ovmr_rg = spl(x_ref)

        spl = spline(x, o2vmr)
        o2vmr_rg = spl(x_ref)

        spl = spline(x, co2vmr)
        co2vmr_rg = spl(x_ref)

        spl = spline(x, n2vmr)
        n2vmr_rg = spl(x_ref)

        ### NOW! call recformula with fomi alpha
        lamb = npl.calc_lamb(pres_rg, temp_rg, ovmr_rg, o2vmr_rg, n2vmr_rg)
        MM = npl.calc_MM(ovmr_rg, o2vmr_rg, n2vmr_rg)

        ## L_esc?
        L_fom = all_coeffs_nlte[('mle', 3, 'l_esc_fom')]
        L_all = npl.coeff_from_interp_lin(interp_coeffs[('Lesc', 'int_fun')], co2vmr_rg)
        uco2 = interp_coeffs['uco2']
        alts_grid = interp_coeffs['alts']

        #Lspl_all = spline(uco2, L_fom, extrapolate = False)
        Lspl_all = spline(uco2, L_all, extrapolate = False)
        uok2 = npl.calc_co2column(alts, pres, temp, co2vmr)

        spl = spline(x, np.log(uok2))
        uok2_rg = spl(x_ref)
        uok2_rg = np.exp(uok2_rg)

        L_esc = Lspl_all(uok2_rg)
        L_esc[np.isnan(L_esc)] = 0.

        alpha_fom = np.array([1.7, 1.6, 1.4, 1.25, 1.14, 1.065]) # Estimate: average of 2 central columns of table 11


        #hr_rec = npl.recformula(alpha_fom, L_esc, lamb, crok, co2vmr_rg, MM, temp_rg, n_alts_trlo = 50, n_alts_trhi = 55, n_alts_cs = 65, ovmr = ovmr_rg)

        alt_fomi, x_fomi, cr_fomi = npl.old_param(alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr, input_in_ppm = False, cart_run_fomi = crun)
        spl = spline(x_fomi, cr_fomi)
        crok = spl(x_ref)
        crfom_new.append(crok)

        #i0 = 49
        i0 = 50

        # Loading exactly fomi alpha and L_esc
        zunk = np.loadtxt(crun + 'debug_alpha__mipas.dat')
        X_fom = zunk[:, 1]
        uco2_fom = zunk[:, 2]

        spl = spline(X_fom, np.exp(zunk[:,3]))
        realpha = spl(x_ref[i0:i0+6])
        # realpha = np.exp(zunk[:6,3])

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

        hr_rec = npl.recformula(realpha, relok, lamb, crok, co2vmr_rg, MM, temp_rg, n_alts_trlo = i0+1, n_alts_trhi = i0+6, n_alts_cs = 65, ovmr = ovmr_rg)
        #hr_rec = npl.recformula(realpha, L_esc, lamb, crok, co2vmr_rg, MM, temp_rg, n_alts_trlo = i0+1, n_alts_trhi = i0+6, n_alts_cs = 65, ovmr = ovmr_rg)

        crfom_recalc.append(hr_rec)
        crfom_old.append(crok)
        xs_ref.append(x_ref)

        co2cols.append(uok2_rg)
        co2cols_fomi.append(uco2_fom)
        # lambs.append(lamb)
        # lescs.append(L_esc)
        # temps.append(temp_rg)
        # press.append(pres_rg)
    ###

    if nmax > 10:
        colors = npl.color_set(10)
        fig = plt.figure()
        for ii, col, cr_fom, cr_rec, x_ref in zip(range(10), colors, crfom_new, crfom_recalc, xs_ref):
            plt.plot(cr_rec, x_ref, color = col, linestyle = '-')
            plt.plot(cr_fom, x_ref, color = col, linestyle = '--')

        plt.ylim(12, None)

        fig = plt.figure()
        for ii, cr_fom, cr_rec, x_ref in zip(range(nmax), crfom_new, crfom_recalc, xs_ref):
            plt.plot(cr_rec-cr_fom, x_ref, color = 'grey', linestyle = '-', lw = 0.1)

        crdiff = np.stack([cr1[:69]-cr2[:69] for cr1, cr2 in zip(crfom_recalc, crfom_new)])
        crme = np.mean(crdiff, axis = 0)
        plt.plot(crme, x_ref[:69], color = 'black')
        crp10 = np.percentile(crdiff, 10, axis = 0)
        crp90 = np.percentile(crdiff, 90, axis = 0)
        plt.plot(crp10, x_ref[:69], color = 'black', ls = '--')
        plt.plot(crp90, x_ref[:69], color = 'black', ls = '--')
        plt.ylabel('x')
        plt.xlabel('hr')
        plt.title('Diff mio - fomichev')
        plt.grid()


#cp = 8.31441e7/MM*(7./2.*(1.-ovmr_rg)+5./2.*ovmr_rg)

###### CHECK single prof

check_single_2 = False
if check_single_2:
    ii = 2

    x = mipx[ii]
    temp = gigi.temperature[ii]
    pres = gigi.pressure[ii]
    ovmr = 1.e-6*O.target[ii]
    co2vmr = 1.e-6*CO2.target[ii]

    # x_ref_max = 16.735
    x_ref = np.arange(0.125, np.max(x) + 0.001, 0.25)

    spl = spline(x, temp)
    temp_rg = spl(x_ref)

    spl = spline(x, np.log(pres))
    pres_rg = spl(x_ref)
    pres_rg = np.exp(pres_rg)

    spl = spline(x, ovmr)
    ovmr_rg = spl(x_ref)

    spl = spline(x, o2vmr)
    o2vmr_rg = spl(x_ref)

    spl = spline(x, co2vmr)
    co2vmr_rg = spl(x_ref)

    spl = spline(x, n2vmr)
    n2vmr_rg = spl(x_ref)

    ### NOW! call recformula with fomi alpha
    lamb = npl.calc_lamb(pres_rg, temp_rg, ovmr_rg, o2vmr_rg, n2vmr_rg)
    MM = npl.calc_MM(ovmr_rg, o2vmr_rg, n2vmr_rg)

    ## L_esc?
    L_fom = all_coeffs_nlte[('mle', 3, 'l_esc_fom')]
    L_all = npl.coeff_from_interp_lin(interp_coeffs[('Lesc', 'int_fun')], co2vmr_rg)
    uco2 = interp_coeffs['uco2']
    alts_grid = interp_coeffs['alts']

    Lspl_all = spline(uco2, L_fom, extrapolate = False)
    #Lspl_all = spline(uco2, L_all, extrapolate = False)
    uok2 = npl.calc_co2column(alts, pres, temp, co2vmr)

    spl = spline(x, np.log(uok2))
    uok2_rg = spl(x_ref)
    uok2_rg = np.exp(uok2_rg)

    L_esc = Lspl_all(uok2_rg)
    L_esc[np.isnan(L_esc)] = 0.

    alpha_fom = np.array([1.7, 1.6, 1.4, 1.25, 1.14, 1.065]) # Estimate: average of 2 central columns of table 11

    spl = spline(x, crfom_ok[ii])
    crok = spl(x_ref)

    hr_rec = npl.recformula(alpha_fom, L_esc, lamb, crok, co2vmr_rg, MM, temp_rg, n_alts_trlo = 50, n_alts_trhi = 55, n_alts_cs = 55)

    alt_fomi2, x_fomi2, cr_fomi2 = npl.old_param(alts, temp, pres, co2vmr, Oprof = ovmr, input_in_ppm = False, cart_run_fomi = '/home/fedef/Research/lavori/CO2_cooling/cart_run_fomi/')
    len(cr_fomi2)
    # plt.figure()
    # plt.plot(cr_fomi2, 1.75 + x_ref[:62])


    zunk = np.loadtxt('/home/fedef/Research/lavori/CO2_cooling/new_param/mipas_check/check_cose_2.dat')
    X_fom = zunk[:, 1]
    T_fom = zunk[:, 2]

    plt.figure()
    plt.plot(T_fom, X_fom, color = 'blue')
    plt.plot(temp_rg, x_ref, color = 'orange', ls = '--')
    plt.title('temp')

    plt.figure()
    plt.plot(zunk[:,5], X_fom, color = 'blue')
    plt.plot(lamb, x_ref, color = 'orange', ls = '--')
    plt.title('lamb')

    plt.figure()
    Lcor = L_esc.copy()
    Lcor[50:56] = alpha_fom*Lcor[50:56]

    plt.plot(zunk[:,6], X_fom, color = 'blue')
    plt.plot(Lcor, x_ref, color = 'orange', ls = '--')
    plt.title('L_esc with alpha (x<14)')

    E_fun = 667.3799
    kbc = 0.69503
    phi_fun = np.exp(-E_fun/(kbc*temp_rg))
    plt.figure()
    plt.plot(zunk[:,7], X_fom, color = 'blue')
    plt.plot(phi_fun, x_ref, color = 'orange', ls = '--')
    plt.title('source funz')

    zunkalp = np.loadtxt('/home/fedef/Research/lavori/CO2_cooling/new_param/mipas_check/check_cose_2_alpha.dat')

    plt.figure()
    plt.plot(zunkalp[:,1], X_fom[:6], color = 'blue')
    plt.plot(uok2_rg, x_ref, color = 'orange', ls = '--')
    plt.title('co2 column')
    plt.xscale('log')

    plt.figure()
    plt.plot(np.exp(zunkalp[:,2]), X_fom[:6], color = 'blue')
    plt.plot(alpha_fom, x_ref[50:56], color = 'orange', ls = '--')
    plt.title('alpha')

    ali = zunk[:,6]
    ali[:6] = np.exp(zunkalp[:,3])
    plt.figure()
    plt.plot(ali, X_fom, color = 'blue')
    plt.plot(L_esc, x_ref, color = 'orange', ls = '--')
    plt.title('L_esc (no corr)')

    plt.figure()
    plt.plot(crfom_ok[ii], X_fom, color = 'blue', linestyle = '-')
    plt.plot(hr_rec, x_ref, color = 'orange', linestyle = '--')
    plt.ylim(12, None)

    plt.figure(6)
    spl = spline(X_fom[:6], np.exp(zunkalp[:,2]))
    realpha = spl(x_ref[49:55])
    plt.plot(realpha, x_ref[49:55], color = 'violet', ls = ':')

    plt.figure(7)
    spl = spline(X_fom, ali)
    reLesc = spl(x_ref[49:49+17])
    plt.plot(reLesc, x_ref[49:49+17], color = 'violet', ls = ':')
    reL = np.zeros(len(L_esc))
    reL[49:49+17] = reLesc
    reL[49+17:] = 1.


    plt.figure(8)
    hr_rec_alphaok = npl.recformula(realpha, L_esc, lamb, crok, co2vmr_rg, MM, temp_rg, n_alts_trlo = 50, n_alts_trhi = 55, n_alts_cs = 55)
    plt.plot(hr_rec_alphaok, x_ref, color = 'red', ls = ':')

    hr_rec_Lok = npl.recformula(realpha, reL, lamb, crok, co2vmr_rg, MM, temp_rg, n_alts_trlo = 50, n_alts_trhi = 55, n_alts_cs = 55)
    plt.plot(hr_rec_Lok, x_ref, color = 'violet', ls = '--')

    hr_rec_Lok_51 = npl.recformula(realpha, reL, lamb, crok, co2vmr_rg, MM, temp_rg, n_alts_trlo = 51, n_alts_trhi = 56, n_alts_cs = 56)
    plt.plot(hr_rec_Lok_51, x_ref, color = 'forestgreen', ls = ':')

#################################################################

check_cr = True
x_ref = np.arange(0.125, 18.01, 0.25)

if check_cr:
    crfom_ok = []
    for x, crmi, crnew, crfom in zip(mipx, gigi.cr_mipas, gigi.cr_new, gigi.cr_fomi):
        # spl = spline(alt_fomi, crfom)
        # crfo2 = spl(alts)
        spl = spline(x_fomi, crfom)
        crfo2 = spl(x)
        crfom_ok.append(crfo2)
    crfom_ok = np.stack(crfom_ok)

    # spaghetti plot
    fig, axs = plt.subplots(1, 2, figsize = (12,8))
    ax1 = axs[0]
    ax2 = axs[1]

    #for crmi, crnew, crfom in zip(gigi.cr_mipas, gigi.cr_new, crfom_ok):
    for x, crmi, crnew, crnew_fa, crfom in zip(mipx, gigi.cr_mipas, new_param, new_param_fa, crfom_ok):
        ax1.plot(-crmi, x, color = 'black', linewidth = 0.1)
        ax1.plot(crfom, x, color = 'blue', linewidth = 0.1, linestyle = '--')
        ax1.plot(crnew, x, color = 'red', linewidth = 0.1, linestyle = '--')
        ax1.plot(crnew_fa, x, color = 'orange', linewidth = 0.1, linestyle = '--')
        ax2.plot(crfom+crmi, x, color = 'blue', linewidth = 0.1, linestyle = '--')
        ax2.plot(crnew+crmi, x, color = 'red', linewidth = 0.1, linestyle = '--')
        ax2.plot(crnew_fa+crmi, x, color = 'orange', linewidth = 0.1, linestyle = '--')

    ax1.set_xlim((-30, 30))
    #ax1.set_ylim((65, 110))
    ax1.set_ylim((10, 18))
    ax2.set_xlim((-30, 30))
    #ax2.set_ylim((65, 110))
    ax2.set_ylim((10, 18))

    fig.savefig(cart_out + 'global_check_spaghetti{}.pdf'.format(version))


    d_fom = []
    d_new = []
    d_new_fa = []
    for x, crmi, crnew, crnew_fa, crfom in zip(mipx, gigi.cr_mipas, new_param, new_param_fa, crfom_ok):
        spl = spline(x, crmi)
        crmi2 = spl(x_ref)
        spl = spline(x, crnew)
        crnew2 = spl(x_ref)
        spl = spline(x, crnew_fa)
        crnew_fa2 = spl(x_ref)
        spl = spline(x, crfom)
        crfom2 = spl(x_ref)

        d_fom.append(crmi2 + crfom2)
        d_new.append(crnew2 + crmi2)
        d_new_fa.append(crnew_fa2 + crmi2)

    d_fom = np.stack(d_fom)
    d_new = np.stack(d_new)
    d_new_fa = np.stack(d_new_fa)
    # d_fom = crfom_ok + np.stack(gigi.cr_mipas)
    # d_new = np.stack(new_param) + np.stack(gigi.cr_mipas)
    # d_new_fa = np.stack(new_param_fa) + np.stack(gigi.cr_mipas)

    ### Figure shading
    fig, ax = plt.subplots()

    dfo_median = np.median(d_fom, axis = 0)
    dfo_1st = np.percentile(d_fom, 25, axis = 0)
    dfo_3rd = np.percentile(d_fom, 75, axis = 0)
    dfo_std = np.std(d_fom, axis = 0)

    dnw_median = np.median(d_new, axis = 0)
    dnw_1st = np.percentile(d_new, 25, axis = 0)
    dnw_3rd = np.percentile(d_new, 75, axis = 0)
    dnw_std = np.std(d_new, axis = 0)

    dnw_median_fa = np.median(d_new_fa, axis = 0)
    dnw_1st_fa = np.percentile(d_new_fa, 25, axis = 0)
    dnw_3rd_fa = np.percentile(d_new_fa, 75, axis = 0)
    dnw_std_fa = np.std(d_new_fa, axis = 0)

    # ax.fill_betweenx(x_ref, dfo_mean-dfo_std, dfo_mean+dfo_std, color = 'blue', alpha = 0.4)
    # ax.fill_betweenx(x_ref, dnw_mean-dnw_std, dnw_mean+dnw_std, color = 'red', alpha = 0.4)
    # ax.plot(dfo_mean, x_ref, color = 'blue', lw = 2)
    # ax.plot(dnw_mean, x_ref, color = 'red', lw = 2)

    ax.fill_betweenx(x_ref, dfo_1st, dfo_3rd, color = 'blue', alpha = 0.4)
    ax.fill_betweenx(x_ref, dnw_1st, dnw_3rd, color = 'red', alpha = 0.4)
    ax.plot(dfo_median, x_ref, color = 'blue', lw = 2)
    ax.plot(dnw_median, x_ref, color = 'red', lw = 2)
    ax.fill_betweenx(x_ref, dnw_1st_fa, dnw_3rd_fa, color = 'orange', alpha = 0.4)
    ax.plot(dnw_median_fa, x_ref, color = 'orange', lw = 2)

    ax.grid()
    ax.set_xlim(-10., 15.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(10., 18.)

    fig.savefig(cart_out + 'global_check_shading{}.pdf'.format(version))


    #########################################
    fig, axs = plt.subplots(3, 3, figsize = (16, 12))

    lats = np.arange(-90, 91, 20)
    for ax, lat1, lat2 in zip(axs.flatten(), lats[:-1], lats[1:]):
        cond = (gigi.latitude > lat1) & (gigi.latitude <= lat2)

        dfok = d_fom[cond]
        dnok = d_new[cond]
        dnok_fa = d_new_fa[cond]

        dfo_median = np.median(dfok, axis = 0)
        dfo_1st = np.percentile(dfok, 25, axis = 0)
        dfo_3rd = np.percentile(dfok, 75, axis = 0)
        dfo_std = np.std(dfok, axis = 0)

        dnw_median = np.median(dnok, axis = 0)
        dnw_1st = np.percentile(dnok, 25, axis = 0)
        dnw_3rd = np.percentile(dnok, 75, axis = 0)
        dnw_std = np.std(dnok, axis = 0)

        dnw_median_fa = np.median(dnok_fa, axis = 0)
        dnw_1st_fa = np.percentile(dnok_fa, 25, axis = 0)
        dnw_3rd_fa = np.percentile(dnok_fa, 75, axis = 0)
        dnw_std_fa = np.std(dnok_fa, axis = 0)

        ax.fill_betweenx(x_ref, dfo_1st, dfo_3rd, color = 'blue', alpha = 0.4)
        ax.fill_betweenx(x_ref, dnw_1st, dnw_3rd, color = 'red', alpha = 0.4)
        ax.plot(dfo_median, x_ref, color = 'blue', lw = 2)
        ax.plot(dnw_median, x_ref, color = 'red', lw = 2)
        ax.fill_betweenx(x_ref, dnw_1st_fa, dnw_3rd_fa, color = 'orange', alpha = 0.4)
        ax.plot(dnw_median_fa, x_ref, color = 'orange', lw = 2)

        ax.grid()
        ax.set_xlim(-10., 10.)
        if lat2 == 90:
            ax.set_xlim(-15., 25.)
        #ax.set_ylim(40., 110.)
        ax.set_ylim(10., 18.)

        ax.set_title('{} to {}'.format(int(lat1), int(lat2)))


    fig.savefig(cart_out + 'global_check_shading_latbands{}.pdf'.format(version))


sys.exit()

############################

### CHECK EOFS

#%%run reparam_high_v8fin.py

temps = np.stack([atm_pt[(atm, 'temp')][alt2:n_top+1] for atm in allatms])
press = np.stack([atm_pt[(atm, 'pres')][alt2:n_top+1] for atm in allatms])


kbc = 0.69503
E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition
phifunz = np.exp(-E_fun/(kbc*temps))
popup = np.array([phifunz[ii]*cose_upper_atm[(atm, cco2, 'lamb')][alt2:n_top+1]/1.5988 for ii, atm in enumerate(allatms) for cco2 in range(1,8)])

popup_mean = np.mean(popup, axis = 0)
popup_anom = popup-popup_mean

solver_pop = Eof(popup_anom)

fig = plt.figure()
for i in range(4):
    plt.plot(solver_pop.eofs()[i], alts[alt2:n_top+1], label= str(i))
plt.title('popup eofs')
plt.legend()
fig.savefig(cart_out + 'eofs_popup_ref.pdf')

## mipas eof
### THIS IS WRONG!!
### the altitude grid is different
### need an interpolation here before

miptemp = np.stack(gigi.temperature)[:, alt2:n_top+1]
mippres = np.stack(gigi.pressure)[:, alt2:n_top+1]
CO2con = 1.e-6*np.stack(CO2.target)[:, alt2:n_top+1]
Ocon = 1.e-6*np.stack(O.target)[:, alt2:n_top+1]

alt_manuel, mol_vmrs, molist, molnums = sbm.read_input_vmr_man(cart_in + 'gases_120.dat', version = 2)
o2vmr = mol_vmrs['O2']*1.e-6
n2vmr = mol_vmrs['N2']*1.e-6
spl = spline(alt_manuel, o2vmr)
o2vmr_or = spl(alts)[alt2:n_top+1]
spl = spline(alt_manuel, n2vmr)
n2vmr_or = spl(alts)[alt2:n_top+1]

phifunz = np.exp(-E_fun/(kbc*miptemp))
lambs = np.stack(npl.calc_lamb(pr, te, opr, o2vmr_or, n2vmr_or) for te, pr, opr in zip(miptemp, mippres, Ocon))

popup_mip = phifunz*lambs/1.5988
#popup_mean = np.mean(popup, axis = 0) USE ORIGINAL MEAN
popup_anom_mip = popup_mip - popup_mean

solver_pop_mip = Eof(popup_anom_mip)

fig = plt.figure()
for i in range(4):
    plt.plot(solver_pop_mip.eofs()[i], alts[alt2:n_top+1], label= str(i))
for i in range(4, 7):
    plt.plot(solver_pop_mip.eofs()[i], alts[alt2:n_top+1], label= str(i), lw = 0.2, ls = '--')

plt.title('popup eofs mipas')
plt.legend()
fig.savefig(cart_out + 'eofs_popup_mipas.pdf')




## check lamb
# fig = plt.figure()
# cco2 = 3
#
# press = np.stack([atm_pt[(atm, 'pres')][alt2:n_top+1] for atm in allatms])
# o_prof3 = [all_coeffs_nlte[(atm, cco2, 'o_vmr')][alt2:n_top+1] for atm in allatms]
# o2_prof3 = [all_coeffs_nlte[(atm, cco2, 'o2_vmr')][alt2:n_top+1] for atm in allatms]
# co2_prof3 = [all_coeffs_nlte[(atm, cco2, 'co2_vmr')][alt2:n_top+1] for atm in allatms]
# n2_prof3 = [all_coeffs_nlte[(atm, cco2, 'n2_vmr')][alt2:n_top+1] for atm in allatms]
#
# lambs = [npl.calc_lamb(pr, te, opr, o2pr, co2pr) for te, pr, opr, o2pr, co2pr in zip(temps, press, o_prof3, o2_prof3, n2_prof3)]
#
# colors = npl.color_set(6)
# for lin, col in zip(lambs, colors):
#     plt.plot(lin, np.arange(alt2, n_top+1), linestyle = '--', color = col)
#
# lamb_check = [cose_upper_atm[(atm, cco2, 'lamb')][alt2:n_top+1] for atm in allatms]
#
# colors = npl.color_set(6)
# for lin, col in zip(lamb_check, colors):
#     plt.plot(lin, np.arange(alt2, n_top+1), color = col)
#
# fig.savefig(cart_out + 'check_lamb_cco2_3.pdf')
