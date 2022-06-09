#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
import math as mt

from subprocess import call
import pickle
import scipy.io as io
from scipy.interpolate import PchipInterpolator as spline

if os.uname()[1] == 'xaru':
    sys.path.insert(0, '/home/fedef/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fedef/Research/git/pythall/')
    cart_out = '/home/fedef/Research/lavori/CO2_cooling/new_param/mipas_check/'
    cart_base = '/home/fedef/Research/'
elif os.uname()[1] == 'hobbes':
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/mipas_check/'
    cart_base = '/home/fabiano/Research/'
else:
    raise ValueError('Unknown platform {}. Specify paths!'.format(os.uname()[1]))

import spect_base_module as sbm
import spect_classes as spcl

import newparam_lib as npl

if not os.path.exists(cart_out): os.mkdir(cart_out)

cart = cart_base + 'lavori/CO2_cooling/MIPAS_2009/'

savT = io.readsav(cart+'CR20090215_v561/L2_20090215_T_561.0', verbose=True)
#savCR = io.readsav(cart+'CR20090215/L2_20090215_CR-CO2-IR_521.6', verbose=True)
savCR = io.readsav(cart+'CR20090215_v561/L2_20090215_CR-CO2-IR@TLOS_561.0', verbose=True)
savO = io.readsav(cart+'CR20090215_v561/L2_20090215_O_561.0', verbose=True)
savCO2 = io.readsav(cart+'CR20090215_v561/L2_20090215_CO2_561.0', verbose=True)

T = savT.result
CR = savCR.result
O = savO.result
CO2 = savCO2.result

# questo stampa i nomi degli ingressi (help di IDL)
print(CR.dtype.names)

alts = CR.altitude[1]
#alts_O = O.altitude[1]

n_p = 67.0
t_n = 23.0
t_s = -23.0
s_p = -67.0
print(len(CR))

#CR = CR[(T.chi2 < 2.0)] # filter for chi2 < 2.0

CR_np = CR[(CR.latitude > n_p).nonzero()]
CR_sp = CR[(CR.latitude < s_p).nonzero()]
CR_mn = CR[(CR.latitude < n_p) & (CR.latitude > t_n)]
CR_ms = CR[(CR.latitude > s_p) & (CR.latitude < t_s)]
CR_eq = CR[(CR.latitude < t_n) & (CR.latitude > t_s)]

# Creo struttura dei risultati

tipi = [('date', 'O'), ('latitude', '>f4'), ('longitude', '>f4'), ('sza', '>f4'), ('altitude', 'O'), ('pressure', 'O'),
        ('temperature', 'O'), ('cr_mipas', 'O'), ('alt_fomi', 'O'), ('cr_fomi', 'O'), ('cr_fomi_int', 'O'), ('cr_new', 'O')]
res = np.empty(1, dtype = tipi)
res = res.view(np.recarray)

restot = res

#####################################################################

ctag = 'v10-nl0-65'
coeff_file = cart_base + 'lavori/CO2_cooling/new_param/reparam_allatm/coeffs_finale_{}.p'.format(ctag)

interp_coeffs = npl.precalc_interp(coeff_file = coeff_file)

#interp_coeffs_old = npl.precalc_interp(n_top = 65, coeff_file = cart_base + 'lavori/CO2_cooling/new_param/reparam_allatm/coeffs_finale_oldv10.p')

interp_coeffs_old = dict()

n_top = 65
for vfit in ['vf4', 'vf5']:
    for afit in ['a{}s'.format(i) for i in range(5)]:
        for n_top in [57, 60, 63, 65, 67, 70, 75]:
            ctag = '{}-{}-{}'.format(vfit, afit, n_top)
            coeff_file = cart_base + 'lavori/CO2_cooling/new_param/newpar_allatm/coeffs_finale_{}.p'.format(ctag)
            interp_coeffs_old[ctag] = npl.precalc_interp_old(coeff_file = coeff_file, n_top = n_top)

coeffs = pickle.load(open(coeff_file, 'rb'))

cart_out_rep = cart_base + 'lavori/CO2_cooling/new_param/NLTE_reparam/'
alpha_unif, _ = pickle.load(open(cart_out_rep +     'alpha_singleatm_v2_top65.p', 'rb'))
alpha_unif = alpha_unif[2] # for present-day co2

# Prova 1: atmosfera polare media durante un SSW
alt_manuel, mol_vmrs, molist, molnums = sbm.read_input_vmr_man(cart + 'gases_120.dat', version = 2)
alt_manuel = np.linspace(0,120,121)

# for il in range(2):
old_param = []
new_param = []
new_param_fa = []
new_param_fixco2 = []
new_param_noextP = []
new_param_starthigh = []
new_param_alt2_50 = []
new_param_alt2_50_fa = []
new_param_fomilike_50 = []
new_param_fomilike_51 = []

new_param_check = dict()
#nams = ['new_fomilike_51_starth', 'new_fa_starth', 'new_alphaunif_fomiLesc', 'new_alphaunif', 'new_old_vf4-a1', 'new_old_vf5-a1', 'new_old_vf4-a2', 'new_old_vf5-a2', 'new_ax05', 'new_ax07', 'new_amedio', 'new_ax2', 'new_old_vf4-a3', 'new_old_vf5-a3']
# nams = ['new_old_{}-a{}'.format(vf, i) for i in range(5) for vf in ['vf4', 'vf5']]
# for nam in nams:
#     new_param_check[nam] = []

vfit = 'vf5'
afit = 'a0s'
# for vfit in ['vf4', 'vf5']:
#     for n_top in [57, 60, 63, 65, 67, 70, 75]:
#         nam = 'new_astart_{}-{}-{}'.format(vfit, afit, n_top)
#         new_param_check[nam] = []

n_top = 75
for vfit in ['vf4', 'vf5']:
    for afit in ['a{}s'.format(i) for i in range(5)]:
        nam = 'new_astart_{}-{}-{}'.format(vfit, afit, n_top)
        new_param_check[nam] = []

#new_param_check['new_mipfit65'] = []

# new_param_check['new_old_vf5-a1_alphareint'] = []
alpha_mip = pickle.load(open(cart_out + 'alpha_mip_fit.p', 'rb'))

alpha3, alpha4, alpha5 = pickle.load(open(cart_out + 'test_alpha.p', 'rb'))

# x_fomi_ref = np.arange(2., 17.26, 0.25)
# x_ref = np.arange(0.125, 18.01, 0.25)
x_ref = np.arange(0.125, 20.625+0.01, 0.25)
x_fomi_ref = np.arange(2., 25, 0.25)

cart_run_fomi = cart_base + 'lavori/CO2_cooling/cart_run_fomi/'

obs = []
alpha_debug = []
L_esc_debug = []
co2column_debug = []

debug_alphafit = []

alpha_fom = []
Lesc_fom = []
co2col_fom = []

do_calc = True
calc_only_new = True

if do_calc:
    inputs = dict()
    for nam in ['temp', 'pres', 'ovmr', 'co2vmr', 'o2vmr', 'n2vmr', 'cr_mipas', 'x']:
        inputs[nam] = []

    mipx = []
    for il in range(len(CR)):
        print(il)
        temp_or = T.target[il]

        #T.target_ig[il]
        pres_or = T.pressure[il]
        #P.target_ig[il]
        x_or = np.log(1000./pres_or) # SHOULD USE THIS FOR INTERPOLATING INSTEAD?

        Ocon_or = O.target[il]
        splO = spline(O.altitude[il], np.log(Ocon_or), extrapolate = False)
        Ocon = splO(alts)
        Ocon = np.exp(Ocon)

        CO2con_or = CO2.target[il]
        splCO2 = spline(CO2.altitude[il], CO2con_or, extrapolate = False)
        CO2con = splCO2(alts)

        splT = spline(T.altitude[il],temp_or, extrapolate = False)
        temp = splT(alts)

        splP = spline(T.altitude[il],np.log(pres_or), extrapolate = False)
        pres = splP(alts)
        pres = np.exp(pres)

        x = np.log(1000./pres)
        mipx.append(x)

        o2spl = spline(alt_manuel, mol_vmrs['O2'])
        O2con = o2spl(alts)
        n2spl = spline(alt_manuel, mol_vmrs['N2'])
        N2con = n2spl(alts)

        o2vmr = O2con*1.e-6
        n2vmr = N2con*1.e-6
        ovmr = Ocon*1.e-6
        co2vmr = CO2con*1.e-6

        inputs['temp'].append(temp)
        inputs['pres'].append(pres)
        inputs['ovmr'].append(ovmr)
        inputs['co2vmr'].append(co2vmr)
        inputs['o2vmr'].append(o2vmr)
        inputs['n2vmr'].append(n2vmr)
        inputs['cr_mipas'].append(CR.target[il])
        inputs['x'].append(x)

        # filename = cart+'atm_manuel.dat'
        # sbm.scriviinputmanuel(alt_manuel,temp,pres,filename)
        #
        # mol_vmrs['CO2'] = CO2con
        # mol_vmrs['O'] = Ocon
        # filename = cart+'vmr_atm_manuel.dat'
        # sbm.write_input_vmr_man(filename, alt_manuel, mol_vmrs, hit_gas_list = molist, hit_gas_num = molnums, version = 2)
        #
        # call(cart+'./fomi_mipas')
        # nomeout = cart+'output__mipas.dat'
        # alt_fomi, cr_fomi = sbm.leggioutfomi(nomeout)
        if not calc_only_new:
            alt_fomi, x_fomi, cr_fomi = npl.old_param(alts, temp, pres, CO2con, Oprof = Ocon, O2prof = O2con, N2prof = N2con, input_in_ppm = True)

            #######
            # spl = spline(alt_manuel, o2vmr)
            # o2vmr_or = spl(alts_O)
            # spl = spline(alt_manuel, n2vmr)
            # n2vmr_or = spl(alts_O)

            #cr_new, debug = npl.new_param_full_allgrids(temp_or, temp_or[0], pres_or, CO2con_or*1.e-6, Ocon_or*1.e-6, o2vmr_or, n2vmr_or, interp_coeffs = interp_coeffs, debug = True)
            cr_new, debug = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = True)

            debug_alphafit.append(debug['alpha_fit'])
            alpha_debug.append(debug['alpha'])
            L_esc_debug.append(debug['L_esc'])
            co2column_debug.append(debug['co2_column'])

            alpha_fom = np.array([1.68717503, 1.52970568, 1.36024627, 1.18849647, 1.0773977, 1.02616183])
            fomialpha = np.append(alpha_fom, np.ones(9))

            cr_new_fa = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = False, debug_alpha = fomialpha)

            cr_new_noextP = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old, debug = False, extrap_co2col = False)


            co2vmr_ref = coeffs['co2profs'][2]
            cr_new_fixco2 = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = False, debug_co2interp = co2vmr_ref)

            ####
            new_param.append(cr_new)
            new_param_fa.append(cr_new_fa)
            new_param_fixco2.append(cr_new_fixco2)
            new_param_noextP.append(cr_new_noextP)

            res.date[0] = CR.date[il]
            res.latitude[0] = CR.latitude[il]
            res.longitude[0] = CR.longitude[il]
            res.sza[0] = CR.sza[il]
            # res.altitude[0] = CR.altitude[il]
            # res.pressure[0] = CR.pressure[il]
            # res.temperature[0] = CR.temperature[il]
            res.altitude[0] = alts
            res.pressure[0] = pres
            res.temperature[0] = temp

            res.cr_mipas[0] = CR.target[il]
            obs.append(CR.target[il])
            res.alt_fomi[0] = alt_fomi
            res.cr_fomi[0] = cr_fomi
            res.cr_new[0] = cr_new

            splcr = spline(alt_fomi, cr_fomi)
            cr_fom_ok = splcr(res.altitude[0])
            old_param.append(cr_fom_ok)
            res.cr_fomi_int[0] = cr_fom_ok

            restot = np.append(restot,res)

            crmi = CR.target[il]
            cspl = spline(x, crmi)
            crmi_ok = cspl(x_ref)
            alt2 = 51
            starthigh = -crmi_ok[alt2-1]

            cr_new_starthigh = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = False, extrap_co2col = True, debug_starthigh = starthigh, debug_alpha = fomialpha)
            new_param_starthigh.append(cr_new_starthigh)

            cr_new_alt2_50 = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = False, extrap_co2col = True, alt2up = 50, n_top = 64)
            new_param_alt2_50.append(cr_new_alt2_50)

            alpha_fom = np.array([1.68717503, 1.52970568, 1.36024627, 1.18849647, 1.0773977, 1.02616183])
            fomialpha = np.append(alpha_fom, np.ones(9))

            cr_new_alt2_50_fa = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = False, extrap_co2col = True, alt2up = 50, n_top = 64, debug_alpha = fomialpha)
            new_param_alt2_50_fa.append(cr_new_alt2_50_fa)


        alt_fomi, x_fomi, cr_fomi = npl.old_param(alts, temp, pres, CO2con, Oprof = Ocon, O2prof = O2con, N2prof = N2con, input_in_ppm = True)

        i0 = 50
        # Loading exactly fomi alpha and L_esc
        zunk = np.loadtxt(cart_run_fomi + 'debug_alpha__mipas.dat')
        X_fom = zunk[:, 1]

        # alpha
        aspl = spline(X_fom, np.exp(zunk[:,3]))
        realpha = aspl(x_ref[i0-1:i0+6])
        alp = np.append(realpha, np.ones(8))

        # L escape
        ali = np.exp(zunk[:,4]) # with no correction
        lspl = spline(X_fom, ali)
        reLesc = lspl(x_ref[i0-1:i0+17])
        reL = np.zeros(len(x_ref))
        reL[i0-1:i0+17] = reLesc
        reL[i0+17:] = 1.
        reL[reL>1.] = 1.

        # cr_new_fomilike_50 = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = False, extrap_co2col = True, alt2up = 50, n_top = 64, debug_alpha = alp, debug_Lesc = reL)
        # new_param_fomilike_50.append(cr_new_fomilike_50)

        i0 = 51
        realpha = aspl(x_ref[i0-1:i0+6])
        alp = np.append(realpha, np.ones(8))

        # co2 column above
        cospl = spline(X_fom, zunk[:, 2])
        recoco = cospl(x_ref[i0-1:i0+17])
        co2col_tot = np.zeros(len(x_ref))
        co2col_tot[i0-1:i0+17] = recoco

        alpha_fom.append(alp)
        Lesc_fom.append(reL)
        co2col_fom.append(co2col_tot)

        # cr_new_fomilike_51 = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = False, extrap_co2col = True, alt2up = 51, n_top = 65, debug_alpha = alp, debug_Lesc = reL)
        # new_param_fomilike_51.append(cr_new_fomilike_51)

        crmi = CR.target[il]
        cspl = spline(x, crmi)
        crmi_ok = cspl(x_ref)
        alt2 = 51
        starthigh = -crmi_ok[alt2-1]

        # nam = 'new_fomilike_51_starth'
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = False, extrap_co2col = True, alt2up = 51, n_top = 65, debug_alpha = alp, debug_Lesc = reL, debug_starthigh = starthigh)
        # new_param_check[nam].append(cr_new)
        #
        # nam = 'new_fa_starth'
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = False, extrap_co2col = False, alt2up = 51, n_top = 65, debug_alpha = alp, debug_starthigh = starthigh)
        # new_param_check[nam].append(cr_new)

        # nam = 'new_alphaunif_fomiLesc'
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = False, extrap_co2col = False, alt2up = 51, n_top = 65, debug_alpha = alpha_unif, debug_Lesc = reL)
        # new_param_check[nam].append(cr_new)
        #
        # nam = 'new_alphaunif'
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug = False, extrap_co2col = False, alt2up = 51, n_top = 65, debug_alpha = alpha_unif)
        # new_param_check[nam].append(cr_new)

        #     for afit in ['a{}'.format(i) for i in range(5)]:
        # vfit = 'vf5'
        # afit = 'a1'
        # afit = 'a0s'
        # for vfit in ['vf4', 'vf5']:
        #     for n_top in [57, 60, 63, 65, 67, 70, 75]:
        #         nam = 'new_astart_{}-{}-{}'.format(vfit, afit, n_top)
        #         cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old['{}-{}-{}'.format(vfit, afit, n_top)], old_param = True, n_top = n_top)
        #         new_param_check[nam].append(cr_new)

        #for n_top in [57, 60, 63, 65, 67, 70, 75]:
        #for vfit in ['vf4', 'vf5']:
        vfit = 'vf5'
        for afit in ['a{}s'.format(i) for i in range(5)]:
            nam = 'new_astart_{}-{}-75'.format(vfit, afit, n_top)
            cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old['{}-{}-{}'.format(vfit, afit, n_top)], old_param = True, n_top = n_top)
            new_param_check[nam].append(cr_new)

        # vfit = 'vf5'
        # afit = 'a1'
        # n_top = 65
        # nam = 'new_mipfit65'
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old['{}-{}-{}'.format(vfit, afit, n_top)], old_param = True, n_top = n_top, debug_alpha = alpha_mip[65])
        # new_param_check[nam].append(cr_new)

        # for vfit in ['vf4', 'vf5']:
        #     for afit in ['a{}'.format(i) for i in range(5)]:
        # vfit = 'vf5'
        # afit = 'a1'
        # nam = 'new_old_{}-{}_alphareint'.format(vfit, afit)
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old['{}-{}-65'.format(vfit, afit)], old_param = True)
        # new_param_check[nam].append(cr_new)

        # nam = 'new_old_vf4-a1'
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old['vf4-a1'], old_param = True)
        # new_param_check[nam].append(cr_new)

        # nam = 'new_old_vf5-a1'
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old_vf5a1, old_param = True)
        # new_param_check[nam].append(cr_new)

        # nam = 'new_old_vf5-a2'
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old_vf5a2, old_param = True)
        # new_param_check[nam].append(cr_new)

        # nam = 'new_ax05' # alpha5
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old_vf5a1, old_param = True, debug_alpha = alpha5)
        # new_param_check[nam].append(cr_new)
        #
        # nam = 'new_ax07' # alpha5
        # alpha = 1+(alpha3-1)*0.7
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old_vf5a1, old_param = True, debug_alpha = alpha5)
        # new_param_check[nam].append(cr_new)
        #
        # nam = 'new_amedio' # alpha3
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old_vf5a1, old_param = True, debug_alpha = alpha3)
        # new_param_check[nam].append(cr_new)
        #
        # nam = 'new_ax2' # alpha4
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old_vf5a1, old_param = True, debug_alpha = alpha4)
        # new_param_check[nam].append(cr_new)

        # nam = 'new_old_vf4-a3'
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old_vf4a3, old_param = True)
        # new_param_check[nam].append(cr_new)
        #
        # nam = 'new_old_vf5-a3'
        # cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old_vf5a3, old_param = True)
        # new_param_check[nam].append(cr_new)


    if not calc_only_new:
        for nam in ['temp', 'pres', 'ovmr', 'co2vmr', 'o2vmr', 'n2vmr', 'cr_mipas', 'x']:
            inputs[nam] = np.array(inputs[nam])

        restot = restot[1:]
        restot = restot.view(np.recarray)

        pickle.dump(restot, open(cart_out+'ssw2009_{}.p'.format(ctag),'wb'))
        pickle.dump([obs, old_param, new_param, new_param_fa, new_param_fixco2, new_param_noextP], open(cart_out+'out_ssw2009_{}.p'.format(ctag),'wb'))
        pickle.dump(inputs, open(cart_out+'in_ssw2009_{}.p'.format(ctag),'wb'))
        pickle.dump([alpha_debug, L_esc_debug, co2column_debug, debug_alphafit], open(cart_out+'debug_ssw2009_{}.p'.format(ctag),'wb'))
        pickle.dump([new_param_starthigh, new_param_alt2_50], open(cart_out+'check_starthigh_out_ssw2009_{}.p'.format(ctag),'wb'))
        pickle.dump(new_param_alt2_50_fa, open(cart_out+'check_alt2_50_fa_out_ssw2009_{}.p'.format(ctag),'wb'))
        pickle.dump([new_param_fomilike_50, new_param_fomilike_51], open(cart_out+'check_fomilike_out_ssw2009_{}.p'.format(ctag),'wb'))
    else:
        #pickle.dump([alpha_fom, Lesc_fom, co2col_fom], open(cart_out+'alpha_Lesc_fom_ssw2009_{}.p'.format(ctag),'wb'))

        #ctag = 'v10-nl0-65'
        ctag = 'vf4-a1'
        new_param_check_old = pickle.load( open(cart_out+'check_all_out_ssw2009_{}.p'.format(ctag),'rb'))

        ctag = '5alphas'
        pickle.dump(new_param_check, open(cart_out+'check_all_out_ssw2009_{}.p'.format(ctag),'wb'))

        for ke in list(new_param_check.keys()):
            if len(new_param_check[ke]) > 0:
                new_param_check[ke] = np.stack(new_param_check[ke])
            else:
                del new_param_check[ke]

        for ke in list(new_param_check_old.keys()):
            if ke in new_param_check:
                del new_param_check_old[ke]
                continue

            if len(new_param_check_old[ke]) > 0:
                new_param_check_old[ke] = np.stack(new_param_check_old[ke])
            else:
                del new_param_check_old[ke]

        new_param_check.update(new_param_check_old)

        ctag = 'vf4-a1'
        pickle.dump(new_param_check, open(cart_out+'check_all_out_ssw2009_{}.p'.format(ctag),'wb'))

if not do_calc or calc_only_new:
    ctag = 'v10-nl0-65'

    restot = pickle.load(open(cart_out+'ssw2009_{}.p'.format(ctag),'rb'))
    obs, old_param, new_param, new_param_fa, new_param_fixco2, new_param_noextP = pickle.load(open(cart_out+'out_ssw2009_{}.p'.format(ctag),'rb'))
    inputs = pickle.load(open(cart_out+'in_ssw2009_{}.p'.format(ctag),'rb'))
    alpha_debug, L_esc_debug, co2column_debug, debug_alphafit = pickle.load(open(cart_out+'debug_ssw2009_{}.p'.format(ctag),'rb'))
    new_param_starthigh, new_param_alt2_50 = pickle.load(open(cart_out+'check_starthigh_out_ssw2009_{}.p'.format(ctag),'rb'))
    new_param_alt2_50_fa = pickle.load(open(cart_out+'check_alt2_50_fa_out_ssw2009_{}.p'.format(ctag),'rb'))

    new_param_fomilike_50, new_param_fomilike_51 = pickle.load(open(cart_out+'check_fomilike_out_ssw2009_{}.p'.format(ctag),'rb'))

    alpha_fom, Lesc_fom, co2col_fom = pickle.load(open(cart_out+'alpha_Lesc_fom_ssw2009_{}.p'.format(ctag),'rb'))

    ctag = 'vf4-a1'
    new_param_check = pickle.load(open(cart_out+'check_all_out_ssw2009_{}.p'.format(ctag),'rb'))


for co, nam in zip([obs, old_param, new_param, new_param_fa, new_param_fixco2, new_param_noextP, new_param_starthigh, new_param_alt2_50, new_param_alt2_50_fa, new_param_fomilike_50, new_param_fomilike_51], ['obs', 'fomi', 'new', 'new_fa', 'new_fixco2', 'new_noextP', 'new_starthigh', 'new_alt2_50', 'new_alt2_50_fa', 'new_fomilike_50', 'new_fomilike_51']):
    new_param_check[nam] = np.stack(co)


# # produco atmosfera di input in formato manuel ->
# # lancio fomi
# # leggo output e salvo in una struttura
# d_fom = []
# d_new = []
# d_new_fa = []
#
# for x, crmi, crnew, crnew_fa, crfom in zip(mipx, restot.cr_mipas, new_param, new_param_fa, old_param):
#     spl = spline(x, crmi)
#     crmi2 = spl(x_ref)
#     spl = spline(x, crnew)
#     crnew2 = spl(x_ref)
#     spl = spline(x, crnew_fa)
#     crnew_fa2 = spl(x_ref)
#     spl = spline(x, crfom)
#     crfom2 = spl(x_ref)
#
#     d_fom.append(crmi2 + crfom2)
#     d_new.append(crnew2 + crmi2)
#     d_new_fa.append(crnew_fa2 + crmi2)
#
# d_fom = np.stack(d_fom)
# d_new = np.stack(d_new)
# d_new_fa = np.stack(d_new_fa)

try:
    crall_rg = pickle.load(open(cart_out + 'crall_rg.p', 'rb'))
except:
    crall_rg = dict()

for na in new_param_check:
    if na in crall_rg:
        continue

    print(na)
    crall_rg[na] = []
    for x, cr in zip(inputs['x'], new_param_check[na]):
        spl = spline(x, cr, extrapolate = False)
        crok = spl(x_ref)
        crall_rg[na].append(crok)

    crall_rg[na] = np.stack(crall_rg[na])

pickle.dump(crall_rg, open(cart_out + 'crall_rg.p', 'wb'))

cco2 = 3

atm_pt = pickle.load(open(cart_out + '../LTE/atm_pt_v4.p'))
all_coeffs_nlte = pickle.load(open(cart_out + '../NLTE/all_coeffs_NLTE.p', 'rb'))
cose_upper_atm = pickle.load(open(cart_out + '../NLTE_upper/cose_upper_atm.p', 'rb'))
all_alts = atm_pt[('mle', 'alts')]


def plot_all_mipas(figtag, nams, colors, dolls = None):
    d_all = dict()
    rms_all = dict()
    d_stats = dict()

    if dolls is None:
        dolls = [True]*len(nams)

    fig, ax = plt.subplots()

    for na, col, doll in zip(nams, colors, dolls):
        co = crall_rg[na] + crall_rg['obs']
        d_all[na] = co

        d_stats[(na, 'median')] = np.nanmedian(co, axis = 0)
        d_stats[(na, '1st')] = np.nanpercentile(co, 25, axis = 0)
        d_stats[(na, '3rd')] = np.nanpercentile(co, 75, axis = 0)
        d_stats[(na, 'std')] = np.nanstd(co, axis = 0)
        d_stats[(na, 'mean')] = np.nanmean(co, axis = 0)

        if doll:
            ax.fill_betweenx(x_ref, d_stats[(na, '1st')], d_stats[(na, '3rd')], color = col, alpha = 0.2)
        # ax.plot(d_stats[(na, '1st')], x_ref, color = col, ls = '--')
        # ax.plot(d_stats[(na, '3rd')], x_ref, color = col, ls = '--')
        ax.plot(d_stats[(na, 'mean')], x_ref, color = col, lw = 2, label = na)

    ax.grid()
    ax.set_xlim(-10., 10.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.legend()

    fig.savefig(cart_out + 'gcs_{}.pdf'.format(figtag))
    #fig.savefig(cart_out + 'gcs_newold_alphasens.pdf')

    #########################################
    fig, axs = plt.subplots(2, 3, figsize = (16, 9))

    lats = np.arange(-90, 91, 30)
    for ax, lat1, lat2 in zip(axs.flatten(), lats[:-1], lats[1:]):
        cond = (restot.latitude > lat1) & (restot.latitude <= lat2)

        for na, col, doll in zip(nams, colors, dolls):
            co = d_all[na][cond]

            d_stats[(na, 'median')] = np.nanmedian(co, axis = 0)
            d_stats[(na, '1st')] = np.nanpercentile(co, 25, axis = 0)
            d_stats[(na, '3rd')] = np.nanpercentile(co, 75, axis = 0)
            d_stats[(na, 'std')] = np.nanstd(co, axis = 0)
            d_stats[(na, 'mean')] = np.nanmean(co, axis = 0)

            if doll:
                ax.fill_betweenx(x_ref, d_stats[(na, '1st')], d_stats[(na, '3rd')], color = col, alpha = 0.2)
            # ax.plot(d_stats[(na, '1st')], x_ref, color = col, ls = '--')
            # ax.plot(d_stats[(na, '3rd')], x_ref, color = col, ls = '--')
            ax.plot(d_stats[(na, 'mean')], x_ref, color = col, lw = 2, label = na)

        ax.grid()
        ax.set_xlim(-10., 10.)
        if lat2 == 90:
            ax.set_xlim(-15., 25.)
        #ax.set_ylim(40., 110.)
        ax.set_ylim(9., 18.)

        ax.set_title('{} to {}'.format(int(lat1), int(lat2)))

    fig.savefig(cart_out + 'gcs_latbands_{}.pdf'.format(figtag))

    fig, ax = plt.subplots()

    for na, col, doll in zip(nams, colors, dolls):
        co = crall_rg[na] + crall_rg['obs']
        coso = np.mean(co**2, axis = 0)
        rms_all[na] = np.sqrt(coso)

        ax.plot(rms_all[na], x_ref, color = col, lw = 2, label = na)

    ax.grid()
    ax.set_xlim(0., 15.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.legend()

    fig.savefig(cart_out + 'gcs_RMS_{}.pdf'.format(figtag))
    #fig.savefig(cart_out + 'gcs_newold_alphasens.pdf')

    #########################################
    fig, axs = plt.subplots(2, 3, figsize = (16, 9))

    lats = np.arange(-90, 91, 30)
    for ax, lat1, lat2 in zip(axs.flatten(), lats[:-1], lats[1:]):
        cond = (restot.latitude > lat1) & (restot.latitude <= lat2)

        for na, col, doll in zip(nams, colors, dolls):
            co = d_all[na][cond]
            coso = np.mean(co**2, axis = 0)
            coso = np.sqrt(coso)

            ax.plot(coso, x_ref, color = col, lw = 2, label = na)

        ax.grid()
        ax.set_xlim(0., 15.)
        if lat2 == 90:
            ax.set_xlim(0., 25.)
        #ax.set_ylim(40., 110.)
        ax.set_ylim(9., 18.)

        ax.set_title('{} to {}'.format(int(lat1), int(lat2)))

    fig.savefig(cart_out + 'gcs_RMS_latbands_{}.pdf'.format(figtag))

    return


def calc_all_refs(cco2 = 3, n_top = 65, debug_alpha = None, interp_coeffs = interp_coeffs_old[('vf5-a0s-65')], use_fomi = False):
    """
    Calcs difference to all reference atms.
    """
    ref_calcs = dict()

    for atm in allatms:
        #calc
        ii = allatms.index(atm)

        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]
        pres = atm_pt[(atm, 'pres')]

        hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_ref')]

        x_ref = np.log(1000./pres)

        co2vmr = atm_pt[(atm, cco2, 'co2')]
        ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
        o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
        n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]

        L_esc = cose_upper_atm[(atm, cco2, 'L_esc_all_extP')]
        lamb = cose_upper_atm[(atm, cco2, 'lamb')]
        MM = cose_upper_atm[(atm, cco2, 'MM')]

        if use_fomi:
            alt_fomi, x_fomi, cr_fomi = npl.old_param(all_alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr, input_in_ppm = False, cart_run_fomi = cart_run_fomi)
            spl = spline(x_fomi, cr_fomi)
            hr_calc = spl(x_ref)
        else:
            hr_calc = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, old_param = True, debug_alpha = debug_alpha, n_top = n_top)

        ref_calcs[atm] = hr_calc-all_coeffs_nlte[(atm, 3, 'hr_ref')]

    return ref_calcs



### now the plots

# figtag = 'vf5-a1-allntops'
# nams = ['fomi']+ ['new_old_vf5-a1-{}'.format(n_top) for n_top in [60, 63, 65, 67, 70]]
# colors = ['blue', 'red', 'gold', 'grey', 'forestgreen', 'violet']
# dolls =  [True, True, False, False, False, False]
# plot_all_mipas(figtag, nams, colors, dolls)
#
# for vf in ['vf4', 'vf5']:
#     figtag = '{}-5alp'.format(vf)
#     nams = ['fomi']+ ['new_old_{}-a{}'.format(vf, i) for i in range(5)]
#     colors = ['blue', 'red', 'gold', 'grey', 'forestgreen', 'violet']
#     dolls =  [True, False, True, False, False, False]
#     plot_all_mipas(figtag, nams, colors, dolls)

# figtag = 'mipfit65'
# nams = ['fomi', 'new_old_vf5-a1-65', 'new_mipfit65']
# colors = ['blue', 'red', 'gold', 'grey', 'forestgreen', 'violet']
# dolls =  [True, True, True, False, False, False]
# plot_all_mipas(figtag, nams, colors, dolls)

figtag = 'astart_vf5_allafit'
nams = ['fomi']+ ['new_astart_{}-a{}s-'.format(vf, i) for i in range(5)]
colors = ['blue', 'red', 'gold', 'grey', 'forestgreen', 'violet']
dolls =  [True, False, True, False, False, False]
plot_all_mipas(figtag, nams, colors, dolls)

figtag = 'astart_vf5_allntops'
vfit = 'vf5'
nams = ['fomi'] + ['new_astart_{}-a0s-{}'.format(vfit, nto) for nto in [57, 60, 63, 65, 67, 70, 75]]
colors = ['blue'] + npl.color_set(7)#, 'red', 'gold', 'grey', 'forestgreen', 'violet']
dolls =  [True, True, False, False, False, False, False, True]
plot_all_mipas(figtag, nams, colors, dolls)


figtag = 'astart_vf4_allntops'
vfit = 'vf4'
nams = ['fomi'] + ['new_astart_{}-a0s-{}'.format(vfit, nto) for nto in [57, 60, 63, 65, 67, 70, 75]]
colors = ['blue'] + npl.color_set(7)#, 'red', 'gold', 'grey', 'forestgreen', 'violet']
dolls =  [True, True, False, False, False, False, False, True]
plot_all_mipas(figtag, nams, colors, dolls)


figtag = 'astart_vf4-vf5_ntop75'
nams = ['fomi', 'new_astart_vf4-a0s-75', 'new_astart_vf5-a0s-75']
colors = ['blue', 'red', 'gold', 'grey', 'forestgreen', 'violet']
dolls =  [True, True, True, False, False, False, False, True]
plot_all_mipas(figtag, nams, colors, dolls)

figtag = 'vf5_a0vsa0s_ntop65'
nams = ['fomi', 'new_old_vf5-a0', 'new_astart_vf5-a0s-65']
colors = ['blue', 'red', 'gold', 'grey', 'forestgreen', 'violet']
dolls =  [True, True, True, False, False, False, False, True]
plot_all_mipas(figtag, nams, colors, dolls)

sys.exit()

d_all = dict()
rms_all = dict()
d_stats = dict()

### Figure shading
for vf in ['vf4', 'vf5']:
    ctag = '{}-5alp'.format(vf)
    fig, ax = plt.subplots()

    nams = ['fomi']+ ['new_old_{}-a{}'.format(vf, i) for i in range(5)] + ['new_old_vf5-a1_alphareint']
    colors = ['blue', 'red', 'gold', 'grey', 'forestgreen', 'violet'] + ['black']
    dolls =  [True, False, False, False, False, False, True]

    for na, col, doll in zip(nams, colors, dolls):
        co = crall_rg[na] + crall_rg['obs']
        d_all[na] = co

        d_stats[(na, 'median')] = np.nanmedian(co, axis = 0)
        d_stats[(na, '1st')] = np.nanpercentile(co, 25, axis = 0)
        d_stats[(na, '3rd')] = np.nanpercentile(co, 75, axis = 0)
        d_stats[(na, 'std')] = np.nanstd(co, axis = 0)
        d_stats[(na, 'mean')] = np.nanmean(co, axis = 0)

        if doll:
            ax.fill_betweenx(x_ref, d_stats[(na, '1st')], d_stats[(na, '3rd')], color = col, alpha = 0.2)
        # ax.plot(d_stats[(na, '1st')], x_ref, color = col, ls = '--')
        # ax.plot(d_stats[(na, '3rd')], x_ref, color = col, ls = '--')
        ax.plot(d_stats[(na, 'mean')], x_ref, color = col, lw = 2, label = na)

    ax.grid()
    ax.set_xlim(-10., 10.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.legend()

    fig.savefig(cart_out + 'gcs_{}_alphareint.pdf'.format(ctag))
    #fig.savefig(cart_out + 'gcs_newold_alphasens.pdf')

    #########################################
    fig, axs = plt.subplots(2, 3, figsize = (16, 9))

    lats = np.arange(-90, 91, 30)
    for ax, lat1, lat2 in zip(axs.flatten(), lats[:-1], lats[1:]):
        cond = (restot.latitude > lat1) & (restot.latitude <= lat2)

        for na, col, doll in zip(nams, colors, dolls):
            co = d_all[na][cond]

            d_stats[(na, 'median')] = np.nanmedian(co, axis = 0)
            d_stats[(na, '1st')] = np.nanpercentile(co, 25, axis = 0)
            d_stats[(na, '3rd')] = np.nanpercentile(co, 75, axis = 0)
            d_stats[(na, 'std')] = np.nanstd(co, axis = 0)
            d_stats[(na, 'mean')] = np.nanmean(co, axis = 0)

            if doll:
                ax.fill_betweenx(x_ref, d_stats[(na, '1st')], d_stats[(na, '3rd')], color = col, alpha = 0.2)
            # ax.plot(d_stats[(na, '1st')], x_ref, color = col, ls = '--')
            # ax.plot(d_stats[(na, '3rd')], x_ref, color = col, ls = '--')
            ax.plot(d_stats[(na, 'mean')], x_ref, color = col, lw = 2, label = na)

        ax.grid()
        ax.set_xlim(-10., 10.)
        if lat2 == 90:
            ax.set_xlim(-15., 25.)
        #ax.set_ylim(40., 110.)
        ax.set_ylim(9., 18.)

        ax.set_title('{} to {}'.format(int(lat1), int(lat2)))

    fig.savefig(cart_out + 'gcs_latbands_{}_alphareint.pdf'.format(ctag))


    #############################################
    ### now with RMS

    fig, ax = plt.subplots()

    for na, col, doll in zip(nams, colors, dolls):
        co = crall_rg[na] + crall_rg['obs']
        coso = np.mean(co**2, axis = 0)
        rms_all[na] = np.sqrt(coso)

        ax.plot(rms_all[na], x_ref, color = col, lw = 2, label = na)

    ax.grid()
    ax.set_xlim(0., 15.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.legend()

    fig.savefig(cart_out + 'gcs_RMS_{}_alphareint.pdf'.format(ctag))
    #fig.savefig(cart_out + 'gcs_newold_alphasens.pdf')

    #########################################
    fig, axs = plt.subplots(2, 3, figsize = (16, 9))

    lats = np.arange(-90, 91, 30)
    for ax, lat1, lat2 in zip(axs.flatten(), lats[:-1], lats[1:]):
        cond = (restot.latitude > lat1) & (restot.latitude <= lat2)

        for na, col, doll in zip(nams, colors, dolls):
            co = d_all[na][cond]
            coso = np.mean(co**2, axis = 0)
            coso = np.sqrt(coso)

            ax.plot(coso, x_ref, color = col, lw = 2, label = na)

        ax.grid()
        ax.set_xlim(0., 15.)
        if lat2 == 90:
            ax.set_xlim(0., 25.)
        #ax.set_ylim(40., 110.)
        ax.set_ylim(9., 18.)

        ax.set_title('{} to {}'.format(int(lat1), int(lat2)))

    fig.savefig(cart_out + 'gcs_RMS_latbands_{}_alphareint.pdf'.format(ctag))

    ##### now with reference

    fig, ax = plt.subplots()

    nams = ['fomi']+ ['new_old_{}-a{}'.format(vf, i) for i in range(5)]
    colors = ['blue', 'red', 'gold', 'grey', 'forestgreen', 'violet']
    dolls =  [False, True, False, False, False, False]

    ref_calcs = dict()
    allatms = npl.allatms
    atmweights = np.array([0.3, 0.1, 0.1, 0.4, 0.05, 0.05])

    for na, col, doll in zip(nams, colors, dolls):
        # refprofs
        for atm in allatms:
            #calc
            ii = allatms.index(atm)

            temp = atm_pt[(atm, 'temp')]
            surf_temp = atm_pt[(atm, 'surf_temp')]
            pres = atm_pt[(atm, 'pres')]

            hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_ref')]

            x_ref = np.log(1000./pres)

            co2vmr = atm_pt[(atm, cco2, 'co2')]
            ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
            o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
            n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]

            L_esc = cose_upper_atm[(atm, cco2, 'L_esc_all_extP')]
            lamb = cose_upper_atm[(atm, cco2, 'lamb')]
            MM = cose_upper_atm[(atm, cco2, 'MM')]

            if na == 'fomi':
                alt_fomi, x_fomi, cr_fomi = npl.old_param(all_alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr, input_in_ppm = False, cart_run_fomi = cart_run_fomi)
                spl = spline(x_fomi, cr_fomi)
                hr_calc = spl(x_ref)
            else:
                version = na.split('_')[-1] + '-65'
                hr_calc = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old[version], old_param = True)

            ref_calcs[(na, atm)] = hr_calc-all_coeffs_nlte[(atm, 3, 'hr_ref')]

        ref_mean = np.mean([ref_calcs[(na, atm)] for atm in allatms], axis = 0)
        ax.plot(ref_mean, x_ref, color = col, lw = 2, label = na)

    ax.grid()
    ax.set_xlim(-10., 10.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.legend()

    fig.savefig(cart_out + 'gcs_reference_{}_latest.pdf'.format(ctag))
    #fig.savefig(cart_out + 'gcs_newold_alphasens.pdf')

    fig, ax = plt.subplots()
    for na, col in zip(nams, colors):
        ref_mean = np.average([ref_calcs[(na, atm)] for atm in allatms], axis = 0, weights = atmweights)
        ax.plot(ref_mean, x_ref, color = col, lw = 2, label = na)

    ax.grid()
    ax.set_xlim(-10., 10.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.legend()

    fig.savefig(cart_out + 'gcs_reference_weighted_{}_latest.pdf'.format(ctag))

    fig, ax = plt.subplots()
    for na, col in zip(nams, colors):
        ref_mean = np.sqrt(np.mean([ref_calcs[(na, atm)]**2 for atm in allatms], axis = 0))
        ax.plot(ref_mean, x_ref, color = col, lw = 2, label = na)

    ax.grid()
    ax.set_xlim(0., 15.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.legend()

    fig.savefig(cart_out + 'gcs_reference_RMS_{}_latest.pdf'.format(ctag))

    fig, ax = plt.subplots()
    for na, col in zip(nams, colors):
        ref_mean = np.sqrt(np.average([ref_calcs[(na, atm)]**2 for atm in allatms], axis = 0, weights = atmweights))
        ax.plot(ref_mean, x_ref, color = col, lw = 2, label = na)

    ax.grid()
    ax.set_xlim(0., 15.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.legend()

    fig.savefig(cart_out + 'gcs_reference_weightedRMS_{}_latest.pdf'.format(ctag))

    #########################################
    fig, axs = plt.subplots(2, 3, figsize = (16, 9))

    lats = np.arange(-90, 91, 30)
    # for ax, lat1, lat2 in zip(axs.flatten(), lats[:-1], lats[1:]):
    #     cond = (restot.latitude > lat1) & (restot.latitude <= lat2)
    #for atm in allatms:
    for atm, ax in zip(['sas', 'mls', 'tro', 'mle', 'mlw', 'saw'], axs.flatten()):
        ax.set_title(atm)
        for na, col, doll in zip(nams, colors, dolls):
            ax.plot(ref_calcs[(na, atm)], x_ref, color = col, lw = 2, label = na)

        ax.grid()
        ax.set_xlim(-10., 10.)
        if lat2 == 90:
            ax.set_xlim(-15., 25.)
        #ax.set_ylim(40., 110.)
        ax.set_ylim(9., 18.)

        #ax.set_title('{} to {}'.format(int(lat1), int(lat2)))

    fig.savefig(cart_out + 'gcs_reference_latbands_{}_latest.pdf'.format(ctag))

    fig, axs = plt.subplots(2, 3, figsize = (16, 9))

    for atm, ax in zip(['sas', 'mls', 'tro', 'mle', 'mlw', 'saw'], axs.flatten()):
        ax.set_title(atm)
        for na, col, doll in zip(nams, colors, dolls):
            rms = np.abs(ref_calcs[(na, atm)])
            ax.plot(rms, x_ref, color = col, lw = 2, label = na)

        ax.grid()
        ax.set_xlim(0., 15.)
        if lat2 == 90:
            ax.set_xlim(0., 25.)
        #ax.set_ylim(40., 110.)
        ax.set_ylim(9., 18.)

        #ax.set_title('{} to {}'.format(int(lat1), int(lat2)))

    fig.savefig(cart_out + 'gcs_reference_latbands_RMS_{}_latest.pdf'.format(ctag))

sys.exit()


###
fig, ax = plt.subplots()

for na, col in zip(['new_amedio', 'new_ax2', 'new_ax05', 'new_ax07'], ['red', 'gold', 'grey', 'forestgreen']):
    co = crall_rg[na] + crall_rg['obs']
    d_all[na] = co

    d_stats[(na, 'median')] = np.nanmedian(co, axis = 0)
    d_stats[(na, '1st')] = np.nanpercentile(co, 25, axis = 0)
    d_stats[(na, '3rd')] = np.nanpercentile(co, 75, axis = 0)
    d_stats[(na, 'std')] = np.nanstd(co, axis = 0)
    d_stats[(na, 'mean')] = np.nanmean(co, axis = 0)

    ax.fill_betweenx(x_ref, d_stats[(na, '1st')], d_stats[(na, '3rd')], color = col, alpha = 0.2)
    # ax.plot(d_stats[(na, '1st')], x_ref, color = col, ls = '--')
    # ax.plot(d_stats[(na, '3rd')], x_ref, color = col, ls = '--')
    ax.plot(d_stats[(na, 'mean')], x_ref, color = col, lw = 2, label = na)

ax.grid()
ax.set_xlim(-10., 10.)
#ax.set_ylim(40., 110.)
ax.set_ylim(9., 18.)

ax.legend()

#fig.savefig(cart_out + 'global_check_shading_newold_{}.pdf'.format(ctag))
fig.savefig(cart_out + 'gcs_newold_alphasens.pdf')

#########################################
fig, axs = plt.subplots(2, 3, figsize = (16, 9))

lats = np.arange(-90, 91, 30)
for ax, lat1, lat2 in zip(axs.flatten(), lats[:-1], lats[1:]):
    cond = (restot.latitude > lat1) & (restot.latitude <= lat2)

    #for na, col in zip(['fomi', 'new', 'new_alphaunif', 'new_old_vf4-a2', 'new_old_vf5-a1'], ['blue', 'red', 'gold', 'grey', 'forestgreen']):
    for na, col in zip(['new_amedio', 'new_ax2', 'new_ax05', 'new_ax07'], ['red', 'gold', 'grey', 'forestgreen']):
        co = d_all[na][cond]

        d_stats[(na, 'median')] = np.nanmedian(co, axis = 0)
        d_stats[(na, '1st')] = np.nanpercentile(co, 25, axis = 0)
        d_stats[(na, '3rd')] = np.nanpercentile(co, 75, axis = 0)
        d_stats[(na, 'std')] = np.nanstd(co, axis = 0)
        d_stats[(na, 'mean')] = np.nanmean(co, axis = 0)

        ax.fill_betweenx(x_ref, d_stats[(na, '1st')], d_stats[(na, '3rd')], color = col, alpha = 0.2)
        # ax.plot(d_stats[(na, '1st')], x_ref, color = col, ls = '--')
        # ax.plot(d_stats[(na, '3rd')], x_ref, color = col, ls = '--')
        ax.plot(d_stats[(na, 'mean')], x_ref, color = col, lw = 2, label = na)

    ax.grid()
    ax.set_xlim(-10., 10.)
    if lat2 == 90:
        ax.set_xlim(-15., 25.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.set_title('{} to {}'.format(int(lat1), int(lat2)))

#fig.savefig(cart_out + 'global_check_shading_latbands_newold_{}.pdf'.format(ctag))
fig.savefig(cart_out + 'gcs_newold_latbands_alphasens.pdf')


##############################################################

fig, ax = plt.subplots()

for na, col, doll in zip(['fomi']+ ['new_old_vf4-a{}'.format(i) for i in range(5)], ['blue', 'red', 'gold', 'grey', 'forestgreen', 'violet'], [False, True, False, False, False, False]):
#for na, col in zip(['new_amedio', 'new_ax2', 'new_ax05'], ['red', 'gold', 'grey', 'forestgreen']):
    co = crall_rg[na] + crall_rg['obs']
    d_all[na] = co

    d_stats[(na, 'median')] = np.nanmedian(co, axis = 0)
    d_stats[(na, '1st')] = np.nanpercentile(co, 25, axis = 0)
    d_stats[(na, '3rd')] = np.nanpercentile(co, 75, axis = 0)
    d_stats[(na, 'std')] = np.nanstd(co, axis = 0)
    d_stats[(na, 'mean')] = np.nanmean(co, axis = 0)

    if doll:
        ax.fill_betweenx(x_ref, d_stats[(na, '1st')], d_stats[(na, '3rd')], color = col, alpha = 0.2)
    # ax.plot(d_stats[(na, '1st')], x_ref, color = col, ls = '--')
    # ax.plot(d_stats[(na, '3rd')], x_ref, color = col, ls = '--')
    ax.plot(d_stats[(na, 'mean')], x_ref, color = col, lw = 2, label = na)

ax.grid()
ax.set_xlim(-10., 10.)
#ax.set_ylim(40., 110.)
ax.set_ylim(9., 18.)

ax.legend()

fig.savefig(cart_out + 'gcs_reference_{}_latest.pdf'.format(ctag))
#fig.savefig(cart_out + 'gcs_newold_alphasens.pdf')

#########################################
fig, axs = plt.subplots(2, 3, figsize = (16, 9))

lats = np.arange(-90, 91, 30)
for ax, lat1, lat2 in zip(axs.flatten(), lats[:-1], lats[1:]):
    cond = (restot.latitude > lat1) & (restot.latitude <= lat2)

    #for na, col, doll in zip(['fomi', 'new', 'new_alphaunif', 'new_old_vf4-a2', 'new_old_vf5-a1'], ['blue', 'red', 'gold', 'grey', 'forestgreen'], [True, False, False, False, True]):
    for na, col, doll in zip(['fomi']+ ['new_old_vf4-a{}'.format(i) for i in range(5)], ['blue', 'red', 'gold', 'grey', 'forestgreen', 'violet'], [False, True, False, False, False, False]):
    #for na, col, doll in zip(['fomi', 'new_old_vf4-a2', 'new_old_vf5-a1', 'new_old_vf4-a1', 'new_old_vf5-a2'], ['blue', 'red', 'gold', 'grey', 'forestgreen'], [True, True, False, False, False]):
        co = d_all[na][cond]

        d_stats[(na, 'median')] = np.nanmedian(co, axis = 0)
        d_stats[(na, '1st')] = np.nanpercentile(co, 25, axis = 0)
        d_stats[(na, '3rd')] = np.nanpercentile(co, 75, axis = 0)
        d_stats[(na, 'std')] = np.nanstd(co, axis = 0)
        d_stats[(na, 'mean')] = np.nanmean(co, axis = 0)

        if doll:
            ax.fill_betweenx(x_ref, d_stats[(na, '1st')], d_stats[(na, '3rd')], color = col, alpha = 0.2)
        # ax.plot(d_stats[(na, '1st')], x_ref, color = col, ls = '--')
        # ax.plot(d_stats[(na, '3rd')], x_ref, color = col, ls = '--')
        ax.plot(d_stats[(na, 'mean')], x_ref, color = col, lw = 2, label = na)

    ax.grid()
    ax.set_xlim(-10., 10.)
    if lat2 == 90:
        ax.set_xlim(-15., 25.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.set_title('{} to {}'.format(int(lat1), int(lat2)))

fig.savefig(cart_out + 'gcs_latbands_reference_{}_latest.pdf'.format(ctag))
