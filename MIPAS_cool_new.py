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
elif os.uname()[1] == 'hobbes':
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/mipas_check/'
else:
    raise ValueError('Unknown platform {}. Specify paths!'.format(os.uname()[1]))

import spect_base_module as sbm
import spect_classes as spcl

import newparam_lib as npl

if not os.path.exists(cart_out): os.mkdir(cart_out)

cart = '/home/fabiano/Research/lavori/CO2_cooling/MIPAS_2009/'

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
coeff_file = '/home/fabiano/Research/lavori/CO2_cooling/new_param/reparam_allatm/coeffs_finale_{}.p'.format(ctag)

interp_coeffs = npl.precalc_interp(coeff_file = coeff_file)

# Prova 1: atmosfera polare media durante un SSW
alt_manuel, mol_vmrs, molist, molnums = sbm.read_input_vmr_man(cart + 'gases_120.dat', version = 2)
alt_manuel = np.linspace(0,120,121)

# for il in range(2):
old_param = []
new_param = []
new_param_fa = []

# x_fomi_ref = np.arange(2., 17.26, 0.25)
# x_ref = np.arange(0.125, 18.01, 0.25)
x_ref = np.arange(0.125, 20.625+0.01, 0.25)
x_fomi_ref = np.arange(2., 25, 0.25)

obs = []
alpha_debug = []
L_esc_debug = []
co2column_debug = []

debug_alphafit = []

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

    ####
    new_param.append(cr_new)
    new_param_fa.append(cr_new_fa)

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


for nam in ['temp', 'pres', 'ovmr', 'co2vmr', 'o2vmr', 'n2vmr', 'cr_mipas', 'x']:
    inputs[nam] = np.array(inputs[nam])

restot = restot[1:]
restot = restot.view(np.recarray)

pickle.dump(restot, open(cart_out+'ssw2009_{}.p'.format(ctag),'wb'))

pickle.dump([obs, old_param, new_param], open(cart_out+'out_ssw2009_{}.p'.format(ctag),'wb'))

pickle.dump(inputs, open(cart_out+'in_ssw2009_{}.p'.format(ctag),'wb'))

pickle.dump([alpha_debug, L_esc_debug, co2column_debug, debug_alphafit], open(cart_out+'debug_ssw2009_{}.p'.format(ctag),'wb'))

# produco atmosfera di input in formato manuel ->
# lancio fomi
# leggo output e salvo in una struttura
d_fom = []
d_new = []
d_new_fa = []

for x, crmi, crnew, crnew_fa, crfom in zip(mipx, restot.cr_mipas, new_param, new_param_fa, old_param):
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

fig.savefig(cart_out + 'global_check_shading_{}.pdf'.format(ctag))
