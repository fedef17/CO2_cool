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
    cart_base = '/home/fedef/Research/lavori/CO2_cooling/new_param/'
elif os.uname()[1] == 'hobbes':
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_base = '/home/fabiano/Research/lavori/CO2_cooling/new_param/'
else:
    raise ValueError('Unknown platform {}. Specify paths!'.format(os.uname()[1]))

import spect_base_module as sbm
import spect_classes as spcl

import newparam_lib as npl

cart_out_mip = cart_base + 'mipas_check/'
cart_out_rep = cart_base + 'NLTE_reparam/'

cart_in = cart_base + '../MIPAS_2009/mipas_v8_cr_t_co2_o_2010/'

####################################################################

interp_coeffs_old = dict()

# n_top = 65
# for vfit in ['vf4', 'vf5']:
#     for afit in ['a{}s'.format(i) for i in range(5)]:
#         for n_top in [57, 60, 63, 65, 67, 70, 75]:

vfit = 'vf5'
afit = 'a0s'
n_top = 65

ctag = '{}-{}-{}'.format(vfit, afit, n_top)
coeff_file = cart_base + 'newpar_allatm/coeffs_finale_{}.p'.format(ctag)
interp_coeffs_old[ctag] = npl.precalc_interp_old(coeff_file = coeff_file, n_top = n_top)

interp_coeffs = interp_coeffs_old[ctag]

alt_manuel, mol_vmrs, molist, molnums = sbm.read_input_vmr_man(cart_in + '../gases_120.dat', version = 2)
alt_manuel = np.linspace(0,120,121)

x_ref = np.arange(0.125, 20.625+0.01, 0.25)
x_fomi_ref = np.arange(2., 25, 0.25)

cart_run_fomi = cart_base + '../cart_run_fomi/'

inputs_all = dict()
new_param_check_all = dict()

do_calc = True


def calc_all_mipas(date, version):
    #######################################################################
    ####### READING FILES
    T = io.readsav(cart_in + 'L2_{}_T_{}'.format(date, version), verbose=True).result
    CR = io.readsav(cart_in + 'L2_{}_CR-CO2-IR_{}'.format(date, version), verbose=True).result
    O = io.readsav(cart_in + 'L2_{}_O_{}'.format(date, version), verbose=True).result
    CO2 = io.readsav(cart_in + 'L2_{}_CO2_{}'.format(date, version), verbose=True).result

    # questo stampa i nomi degli ingressi (help di IDL)
    print(CR.dtype.names)
    alts = CR.altitude[1]

    # Creo struttura dei risultati
    tipi = [('date', 'O'), ('latitude', '>f4'), ('longitude', '>f4'), ('sza', '>f4'), ('altitude', 'O'), ('pressure', 'O'),
    ('temperature', 'O'), ('cr_mipas', 'O'), ('alt_fomi', 'O'), ('cr_fomi', 'O'), ('cr_fomi_int', 'O'), ('cr_new', 'O')]
    res = np.empty(1, dtype = tipi)
    res = res.view(np.recarray)
    restot = res
    #####################################################################

    inputs = dict()
    for nam in ['temp', 'pres', 'ovmr', 'co2vmr', 'o2vmr', 'n2vmr', 'cr_mipas', 'x', 'lat', 'lon', 'sza']:
        inputs[nam] = []
    new_param_check = dict()
    for nam in ['obs', 'fomi', 'new_vf5-a0s-65']:
        new_param_check[nam] = []

    for il in range(len(CR)):
        print(il)
        temp_or = T.target[il]
        pres_or = T.pressure[il]
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
        inputs['lat'].append(CR.latitude[il])
        inputs['lon'].append(CR.longitude[il])
        inputs['sza'].append(CR.sza[il])

        alt_fomi, x_fomi, cr_fomi = npl.old_param(alts, temp, pres, CO2con, Oprof = Ocon, O2prof = O2con, N2prof = N2con, input_in_ppm = True, cart_run_fomi = cart_base + '../cart_run_fomi/')
        splcr = spline(alt_fomi, cr_fomi)
        cr_fom_ok = splcr(alts)
        new_param_check['fomi'].append(cr_fom_ok)

        cr_new = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, old_param = True, n_top = n_top)
        new_param_check['new_vf5-a0s-65'].append(cr_new)

    for ke in new_param_check:
        new_param_check[ke] = np.stack(new_param_check[ke])

    for ke in inputs:
        inputs[ke] = np.stack(inputs[ke])

    return inputs, new_param_check


alldates = [co[3:11] for co in os.listdir(cart_in) if 'L2' in co and 'T_' in co]
version = '561.0'

for dat in alldates:
    inp, npc = calc_all_mipas(dat, version)
    inputs_all[(dat, version)] = inp
    new_param_check_all[(dat, version)] = npc

    pickle.dump([inp, npc], open(cart_out_mip + 'mipcalc_{}_{}_{}.p'.format(dat, version, ctag), 'wb'))

    np.savetxt(cart_out_mip + 'crmipas_{}_{}.txt'.format(dat, version), npc['obs'])
    np.savetxt(cart_out_mip + 'crfomi_{}_{}.txt'.format(dat, version), npc['fomi'])
    np.savetxt(cart_out_mip + 'crnew_{}_{}_{}.txt'.format(dat, version, ctag), npc['new_{}'.format(ctag)])

###############################################################################################################

try:
    crall_rg = pickle.load(open(cart_out_mip + 'crall_rg_mipall.p', 'rb'))
except:
    crall_rg = dict()

for cos in new_param_check_all:
    for na in new_param_check[cos]:
        if (cos, na) in crall_rg:
            continue

        print(cos, na)
        crall_rg[(cos, na)] = []
        for x, cr in zip(inputs[cos]['x'], new_param_check[cos][na]):
            spl = spline(x, cr, extrapolate = False)
            crok = spl(x_ref)
            crall_rg[(cos, na)].append(crok)

        crall_rg[(cos, na)] = np.stack(crall_rg[(cos, na)])

    pickle.dump(crall_rg, open(cart_out_mip + 'crall_rg_mipall.p', 'wb'))

#######################################################################################
## now the plots
cco2 = 3
atm_pt = pickle.load(open(cart_base + 'LTE/atm_pt_v4.p'))
all_coeffs_nlte = pickle.load(open(cart_base + 'NLTE/all_coeffs_NLTE.p', 'rb'))
cose_upper_atm = pickle.load(open(cart_base + 'NLTE_upper/cose_upper_atm.p', 'rb'))
all_alts = atm_pt[('mle', 'alts')]


def plot_all_mipas(dat, version, figtag, nams, colors, dolls = None, tags = None):
    d_all = dict()
    rms_all = dict()
    d_stats = dict()

    if dolls is None:
        dolls = [True]*len(nams)

    if tags is None:
        tags = nams

    fig, ax = plt.subplots()

    for na, ta, col, doll in zip(nams, tags, colors, dolls):
        co = crall_rg[(date, version, na)] + crall_rg[(date, version, 'obs')]
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
        ax.plot(d_stats[(na, 'mean')], x_ref, color = col, lw = 2, label = ta)

    ax.grid()
    ax.set_xlim(-10., 10.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.legend(fontsize = 'small')
    ax.set_xlabel('HR (K/day)')
    ax.set_ylabel('x')

    fig.savefig(cart_out_mip + 'gcs_{}.pdf'.format(figtag))
    #fig.savefig(cart_out_mip + 'gcs_newold_alphasens.pdf')

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
        ax.set_xlabel('HR (K/day)')
        ax.set_ylabel('x')

        ax.set_title('{} to {}'.format(int(lat1), int(lat2)))

    fig.savefig(cart_out_mip + 'gcs_latbands_{}.pdf'.format(figtag))

    fig, ax = plt.subplots()

    for na, ta, col, doll in zip(nams, tags, colors, dolls):
        co = d_all[na]
        coso = np.mean(co**2, axis = 0)
        rms_all[na] = np.sqrt(coso)

        ax.plot(rms_all[na], x_ref, color = col, lw = 2, label = ta)

    ax.grid()
    ax.set_xlim(0., 15.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(9., 18.)

    ax.legend(fontsize = 'small')
    ax.set_xlabel('RMS (K/day)')
    ax.set_ylabel('x')

    fig.savefig(cart_out_mip + 'gcs_RMS_{}.pdf'.format(figtag))
    #fig.savefig(cart_out_mip + 'gcs_newold_alphasens.pdf')

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
        ax.set_xlabel('RMS (K/day)')
        ax.set_ylabel('x')

        ax.set_title('{} to {}'.format(int(lat1), int(lat2)))

    fig.savefig(cart_out_mip + 'gcs_RMS_latbands_{}.pdf'.format(figtag))

    return


def calc_all_refs(cco2 = 3, n_top = 65, debug_alpha = None, interp_coeffs = interp_coeffs, use_fomi = False):
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
for dat in alldates:
    vfit = 'vf5'
    figtag = 'mipcalc_{}_{}_{}'.format(dat, version, ctag)
    nams = ['fomi', 'new_vf5-a0s-65']
    colors = ['blue', 'red', 'gold', 'grey', 'forestgreen', 'violet']
    dolls =  [True, True]
    plot_all_mipas(dat, version, figtag, nams, colors, dolls = dolls, tags = tags)
