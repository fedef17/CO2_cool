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
    cart_base = '/home/fedef/Research/lavori/CO2_cooling/new_param/'
elif os.uname()[1] == 'hobbes':
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_base = '/home/fabiano/Research/lavori/CO2_cooling/new_param/'

import spect_base_module as sbm

############################

cart_out = cart_base + 'LTE/'
cart_out_2 = cart_base + 'NLTE/'
cart_out_rep = cart_base + 'NLTE_reparam/'
cart_out_3 = cart_base + 'NLTE_upper/'
cart_out_F = cart_base + 'newpar_allatm_2/'

allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
#atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweights = np.ones(6)/6.
atmweights = dict(zip(allatms, atmweights))
allco2 = np.arange(1,npl.n_co2prof+1)

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v4.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt_v4.p'))

all_alts = atm_pt[('mle', 'alts')]

pres = atm_pt[('mle', 'pres')]
x_ref = np.log(1000./pres)

all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

cose_upper_atm = pickle.load(open(cart_out_3 + 'cose_upper_atm.p', 'rb'))

alt2 = 51
n_top = 65
############################

cart_out = cart_base + 'sens_test/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

cart_in = cart_base + 'sent10_co2interm_hrnlte_ngt_co2inter_kod2/'

#####################################################
datain = dict()
# co2 intermedi: cr_nlte_saw_co2_9.sav
filsav = 'cr/cr_nlte_{}_co2_{}.sav'
for cco2 in range(9, 12):
    for atm in allatms:
        coso = io.readsav(cart_in+filsav.format(atm, cco2))['data']
        nomi = 'HR_NLTE HR_NLTE_FB HR_NLTE_HOT HR_NLTE_ISO HR_LTE CO2_VMR O_VMR UCO2 L_ESC L_ESC_FOM O2_VMR N2_VMR'
        nomi = nomi.split()

        for nom in nomi:
            vara = getattr(coso, nom)[0]
            if 'HR' in nom:
                datain[(atm, cco2, nom.lower())] = -vara
            else:
                datain[(atm, cco2, nom.lower())] = vara

# Ko dimezzato: cr_nlte_mle_co2_3_kod2.sav
filsav = 'cr/cr_nlte_{}_co2_{}_kod2.sav'
cco2 = 3
for atm in allatms:
    coso = io.readsav(cart_in+filsav.format(atm, cco2))['data']
    nomi = 'HR_NLTE HR_NLTE_FB HR_NLTE_HOT HR_NLTE_ISO HR_LTE CO2_VMR O_VMR UCO2 L_ESC L_ESC_FOM O2_VMR N2_VMR'
    nomi = nomi.split()

    for nom in nomi:
        vara = getattr(coso, nom)[0]
        if 'HR' in nom:
            datain[(atm, cco2, nom.lower(), 'kod2')] = -vara
        else:
            datain[(atm, cco2, nom.lower(), 'kod2')] = vara

# prf file not needed: the profiles are the same, just at higher resolution
# filvmr = 'co2_vmr/vmr_cira_mle_co2_10.prf'
# atmalts, mol_vmr, _, _ = sbm.read_input_vmr_man(cart_in + filvmr)

#####################################################

######################################################################
#################### SCELTA SET PARAMETRI! ###########################
######################################################################

vfit = 'vf5'
afit = 'a0s'
n_top = 65

ctag = '{}-{}-{}'.format(vfit, afit, n_top)
coeff_file = cart_base + 'newpar_allatm/coeffs_finale_{}.p'.format(ctag)

interp_coeffs = npl.precalc_interp_old(coeff_file = coeff_file, n_top = n_top)

######################################################################
#################### SCELTA SET PARAMETRI! ###########################
######################################################################

cart_run_fomi = '/home/{}/Research/lavori/CO2_cooling/cart_run_fomi/'.format(os.getlogin())

figs = []
calcs = dict()

# for cco2 in range(9, 12):
#     refcalcs = []
#     for atm in allatms:
#         #calc
#         ii = allatms.index(atm)
#
#         temp = atm_pt[(atm, 'temp')]
#         surf_temp = atm_pt[(atm, 'surf_temp')]
#         pres = atm_pt[(atm, 'pres')]
#
#         hr_ref = datain[(atm, cco2, 'hr_nlte')]
#
#         x_ref = np.log(1000./pres)
#
#         co2vmr = datain[(atm, cco2, 'co2_vmr')]
#         ovmr = datain[(atm, cco2, 'o_vmr')]
#         o2vmr = datain[(atm, cco2, 'o2_vmr')]
#         n2vmr = datain[(atm, cco2, 'n2_vmr')]
#
#         alt_fomi, x_fomi, cr_fomi = npl.old_param(all_alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr, input_in_ppm = False, cart_run_fomi = cart_run_fomi)
#         spl = spline(x_fomi, cr_fomi)
#         hr_calc = spl(x_ref)
#
#         calcs[(atm, cco2, 'fomi')] = hr_calc
#
#         res = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, old_param = True, n_top = n_top, extrap_co2col = True)
#
#         calcs[(atm, cco2)] = res
#
#         fig, a0, a1 = npl.manuel_plot(x_ref, [hr_ref, res, hr_calc], ['ref', 'new','fomi'], xlabel = 'HR (K/day)', ylabel = 'X', title = '{} - {}'.format(atm, cco2), xlimdiff = None, xlim = None, ylim = (9.5, 21), linestyles = ['-', '--', ':'], colors = ['black', 'red', 'blue'], orizlines = [9.875, 12.625, 16.375, 20.125])
#         figs.append(fig)
#
# #pickle.dump(calcs, open(cart_out + 'calcs_interm.p', 'wb'))
#
# ###### write output in txt
# for cco2 in range(9,12):
#     refcal = np.stack([calcs[(atm, cco2)] for atm in allatms])
#     np.savetxt(cart_out + 'param_interm_co2-{}.txt'.format(cco2), refcal)
#
#     refcal = np.stack([calcs[(atm, cco2, 'fomi')] for atm in allatms])
#     np.savetxt(cart_out + 'fomi_interm_co2-{}.txt'.format(cco2), refcal)
#
#     refhr = np.stack([datain[(atm, cco2, 'hr_nlte')] for atm in allatms])
#     np.savetxt(cart_out + 'ref_interm_co2-{}.txt'.format(cco2), refhr)


###################################################
# for ko/2
from subprocess import call

cco2 = 3
for atm in allatms:
    #calc
    ii = allatms.index(atm)

    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]
    pres = atm_pt[(atm, 'pres')]

    hr_ref = datain[(atm, cco2, 'hr_nlte', 'kod2')]

    x_ref = np.log(1000./pres)

    co2vmr = datain[(atm, cco2, 'co2_vmr', 'kod2')]
    ovmr = datain[(atm, cco2, 'o_vmr', 'kod2')] # divide by 2!
    o2vmr = datain[(atm, cco2, 'o2_vmr', 'kod2')]
    n2vmr = datain[(atm, cco2, 'n2_vmr', 'kod2')]


    alt_fomi, x_fomi, cr_fomi = npl.old_param(all_alts, temp, pres, co2vmr, Oprof = ovmr/2., O2prof = o2vmr, N2prof = n2vmr, input_in_ppm = False, cart_run_fomi = cart_run_fomi)
    spl = spline(x_fomi, cr_fomi)
    hr_calc = spl(x_ref)

    calcs[(atm, cco2, 'fomi', 'kod2')] = hr_calc

    wd = os.getcwd()
    os.chdir(cart_run_fomi)
    call('cp input_atm_kod2.dat input_atm.dat')
    os.chdir(wd)
    alt_fomi, x_fomi, cr_fomi = npl.old_param(all_alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr, input_in_ppm = False, cart_run_fomi = cart_run_fomi)
    spl = spline(x_fomi, cr_fomi)
    hr_calc = spl(x_ref)
    os.chdir(cart_run_fomi)
    call('cp input_atm_default.dat input_atm.dat')
    os.chdir(wd)

    calcs[(atm, cco2, 'fomi', 'zofac')] = hr_calc

    res = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr/2., o2vmr, n2vmr, interp_coeffs = interp_coeffs, old_param = True, n_top = n_top, extrap_co2col = True)

    calcs[(atm, cco2, 'kod2')] = res

    res2 = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, old_param = True, n_top = n_top, extrap_co2col = True, zofac = 0.5)

    calcs[(atm, cco2, 'zofac')] = res2

    fig, a0, a1 = npl.manuel_plot(x_ref, [hr_ref, res2, hr_calc], ['ref', 'ZOdiv2', 'fomi'], xlabel = 'HR (K/day)', ylabel = 'X', title = '{} - {} - KOD2'.format(atm, cco2), xlimdiff = None, xlim = None, ylim = (9.5, 21), linestyles = ['-', '--', ':', ':'], colors = ['black', 'red', 'orange', 'blue'], orizlines = [9.875, 12.625, 16.375, 20.125])
    figs.append(fig)

npl.plot_pdfpages(cart_out + 'check_calc_kod2.pdf', figs)


pickle.dump(calcs, open(cart_out + 'calcs_interm.p', 'wb'))

###### write output in txt
refcal = np.stack([calcs[(atm, cco2, 'kod2')] for atm in allatms])
np.savetxt(cart_out + 'param_Odiv2_co2-{}.txt'.format(cco2), refcal)

refcal = np.stack([calcs[(atm, cco2, 'zofac')] for atm in allatms])
np.savetxt(cart_out + 'param_ZOdiv2_co2-{}.txt'.format(cco2), refcal)

refcal = np.stack([calcs[(atm, cco2, 'fomi', 'kod2')] for atm in allatms])
np.savetxt(cart_out + 'fomi_Odiv2_co2-{}.txt'.format(cco2), refcal)

refcal = np.stack([calcs[(atm, cco2, 'fomi', 'zofac')] for atm in allatms])
np.savetxt(cart_out + 'fomi_ZOdiv2_co2-{}.txt'.format(cco2), refcal)


refhr = np.stack([datain[(atm, cco2, 'hr_nlte', 'kod2')] for atm in allatms])
np.savetxt(cart_out + 'ref_kod2_co2-{}.txt'.format(cco2), refhr)
