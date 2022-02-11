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

cart_out_F = cart_base + 'newpar_allatm_2/'

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
################################################################################

### LOADING MIPAS

cart_out = cart_base + 'mipas_check/'
version = '_xinterp_v3'

savT = io.readsav(cart_in+'CR20090215/L2_20090215_T_521.6', verbose=True)
savCR = io.readsav(cart_in+'CR20090215/L2_20090215_CR-CO2-IR_521.6', verbose=True)
savO = io.readsav(cart_in+'CR20090215/L2_20090215_O_521.6', verbose=True)
savCO2 = io.readsav(cart_in+'CR20090215/L2_20090215_CO2_521.6', verbose=True)

T = savT.result
CR = savCR.result
O = savO.result
CO2 = savCO2.result

sys.exit()

fil = 'ssw2009_v3_okTOCO2_1e13_newparam{}.p'.format(version)

gigi = pickle.load(open(cart_out+fil, 'rb'))
cose = gigi.dtype.names

obs, old_param, new_param, new_param_fa  = pickle.load(open(cart_out+'out_ssw2009{}.p'.format(version),'rb'))

obs = np.stack(gigi.cr_mipas)
alts = gigi.altitude[0]

alt_manuel, mol_vmrs, molist, molnums = sbm.read_input_vmr_man(cart_in + 'gases_120.dat', version = 2)
o2vmr = mol_vmrs['O2']*1.e-6
n2vmr = mol_vmrs['N2']*1.e-6
spl = spline(alt_manuel, o2vmr)
o2vmr = spl(alts)
spl = spline(alt_manuel, n2vmr)
n2vmr = spl(alts)

miptemp = np.stack(gigi.temperature)
mippres = np.stack(gigi.pressure)
mipx = np.log(1000./mippres)
mipovmr = 1.e-6*np.stack(O.target)
mipco2vmr = 1.e-6*np.stack(CO2.target)

######

regrcoef = pickle.load(open(cart_out_rep + 'regrcoef_v3.p', 'rb'))
nlte_corr = pickle.load(open(cart_out_rep + 'nlte_corr_low.p', 'rb'))

alt2 = 50
n_top = 64

x_ref = np.arange(0.125, 18.01, 0.25)

mip_temp_rg = []
mip_pres_rg = []
mip_ovmr_rg = []
mip_co2vmr_rg = []
mip_o2vmr_rg = []
mip_n2vmr_rg = []

for temp, pres, x, co2vmr, ovmr in zip(miptemp, mippres, mipx, mipco2vmr, mipovmr):
    spl = spline(x, temp, extrapolate = False)
    temp_rg = spl(x_ref)

    spl = spline(x, np.log(pres), extrapolate = False)
    pres_rg = spl(x_ref)
    pres_rg = np.exp(pres_rg)

    spl = spline(x, ovmr, extrapolate = False)
    ovmr_rg = spl(x_ref)

    spl = spline(x, co2vmr, extrapolate = False)
    co2vmr_rg = spl(x_ref)

    spl = spline(x, o2vmr, extrapolate = False)
    o2vmr_rg = spl(x_ref)

    spl = spline(x, n2vmr, extrapolate = False)
    n2vmr_rg = spl(x_ref)

    mip_temp_rg.append(temp_rg)
    mip_pres_rg.append(pres_rg)
    mip_ovmr_rg.append(ovmr_rg)
    mip_co2vmr_rg.append(co2vmr_rg)
    mip_o2vmr_rg.append(o2vmr_rg)
    mip_n2vmr_rg.append(n2vmr_rg)

mip_temp_rg = np.stack(mip_temp_rg)
mip_pres_rg = np.stack(mip_pres_rg)
mip_ovmr_rg = np.stack(mip_ovmr_rg)
mip_co2vmr_rg = np.stack(mip_co2vmr_rg)
mip_o2vmr_rg = np.stack(mip_o2vmr_rg)
mip_n2vmr_rg = np.stack(mip_n2vmr_rg)

########### Qui c'è la parte del fit dell'alpha
# alpha FIT!

n_trans = n_top-alt2+1

#bounds = (0.1*np.ones(n_trans), 100*np.ones(n_trans))
bounds = (0.5*np.ones(n_trans), 10*np.ones(n_trans))

mip_alpha = []
mip_alpha_baro = []
start = np.ones(n_trans)

interp_coeffs = npl.precalc_interp()

for ii in range(len(obs)):
    print(ii)
    temp = mip_temp_rg[ii][:n_top+1]
    pres = mip_pres_rg[ii][:n_top+1]
    co2vmr = mip_co2vmr_rg[ii][:n_top+1]
    ovmr = mip_ovmr_rg[ii][:n_top+1]
    o2vmr = mip_o2vmr_rg[ii][:n_top+1]
    n2vmr = mip_n2vmr_rg[ii][:n_top+1]

    MM = npl.calc_MM(ovmr, o2vmr, n2vmr)

    L_all = npl.coeff_from_interp_lin(interp_coeffs[('Lesc', 'int_fun')], co2vmr)
    uco2 = interp_coeffs['uco2']
    Lspl_all = spline(uco2, L_all, extrapolate = False)
    uok = npl.calc_co2column_P(pres, co2vmr, MM)
    L_esc = Lspl_all(uok)
    L_esc[np.isnan(L_esc)] = 0.

    lamb = npl.calc_lamb(pres, temp, ovmr, o2vmr, n2vmr)

    spl = spline(mipx[ii], -obs[ii], extrapolate = False)
    hr_ref = spl(x_ref)

    result = least_squares(npl.delta_alpha_rec3_general, start, args=(hr_ref[alt2], hr_ref[:n_top+1], temp, pres, co2vmr, ovmr, o2vmr, n2vmr, alt2, n_top, interp_coeffs, L_esc, MM, lamb), verbose=1, method = 'trf', bounds = bounds)#, gtol = gtol, xtol = xtol)
    print('least_squares', result)
    mip_alpha.append(result.x)

    ### Stessa cosa ma partendo dal mio HR:
    spl = spline(mipx[ii], new_param[ii], extrapolate = False)
    hr_ref = spl(x_ref)

    result = least_squares(npl.delta_alpha_rec3_general, start, args=(hr_ref[alt2], hr_ref[:n_top+1], temp, pres, co2vmr, ovmr, o2vmr, n2vmr, alt2, n_top, interp_coeffs, L_esc, MM, lamb), verbose=1, method = 'trf', bounds = bounds)#, gtol = gtol, xtol = xtol)
    print('least_squares', result)
    mip_alpha_baro.append(result.x)

mip_alpha = np.stack(mip_alpha)
mip_alpha_baro = np.stack(mip_alpha_baro)
pickle.dump([mip_alpha, mip_alpha_baro], open(cart_out + 'mip_alphas.p', 'wb'))

alpha_mea = np.mean(mip_alpha, axis = 0)
alpha_q1 = np.percentile(mip_alpha, 25, axis = 0)
alpha_q3 = np.percentile(mip_alpha, 75, axis = 0)
alpha_median = np.percentile(mip_alpha, 50, axis = 0)
fig = plt.figure()
x_ref[alt2:n_top+1]
for ii in range(len(mip_alpha)):
    plt.plot(mip_alpha[ii], x_ref[alt2:n_top+1], color = 'grey', lw = 0.1)
plt.plot(alpha_mea,  x_ref[alt2:n_top+1], color = 'red')
plt.plot(alpha_median,  x_ref[alt2:n_top+1], color = 'red', ls = '--')
plt.plot(alpha_q1,  x_ref[alt2:n_top+1], color = 'red', ls = ':')
plt.plot(alpha_q3,  x_ref[alt2:n_top+1], color = 'red', ls = ':')

fig.savefig(cart_out + 'refit_alpha_mipas_05-10.pdf')

alpha_mea = np.mean(mip_alpha_baro-mip_alpha, axis = 0)
alpha_q1 = np.percentile(mip_alpha_baro-mip_alpha, 25, axis = 0)
alpha_q3 = np.percentile(mip_alpha_baro-mip_alpha, 75, axis = 0)
alpha_median = np.percentile(mip_alpha_baro-mip_alpha, 50, axis = 0)
fig = plt.figure()
x_ref[alt2:n_top+1]
for ii in range(len(mip_alpha)):
    plt.plot(mip_alpha_baro[ii]-mip_alpha[ii], x_ref[alt2:n_top+1], color = 'grey', lw = 0.1)
plt.plot(alpha_mea,  x_ref[alt2:n_top+1], color = 'red')
plt.plot(alpha_median,  x_ref[alt2:n_top+1], color = 'red', ls = '--')
plt.plot(alpha_q1,  x_ref[alt2:n_top+1], color = 'red', ls = ':')
plt.plot(alpha_q3,  x_ref[alt2:n_top+1], color = 'red', ls = ':')

fig.savefig(cart_out + 'refit_alpha_mipas_barodiff.pdf')

mip_alpha = mip_alpha_baro

######### CHECK HR
version = '_refit_baro'

fig, axs = plt.subplots(1, 2)

fit_new = []
fit_alpha1 = []

for ii in range(len(obs)):
    print(ii)
    temp = mip_temp_rg[ii][:n_top+2]
    pres = mip_pres_rg[ii][:n_top+2]
    co2vmr = mip_co2vmr_rg[ii][:n_top+2]
    ovmr = mip_ovmr_rg[ii][:n_top+2]
    o2vmr = mip_o2vmr_rg[ii][:n_top+2]
    n2vmr = mip_n2vmr_rg[ii][:n_top+2]

    alp = mip_alpha[ii]

    cr_new_alpha = npl.new_param_full(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug_alpha = alp)
    fit_new.append(cr_new_alpha)

    cr_alpha1 = npl.new_param_full(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, debug_alpha = np.ones(15))
    fit_alpha1.append(cr_alpha1)

    spl = spline(mipx[ii], -obs[ii], extrapolate = False)
    hr_ref = spl(x_ref)

    axs[0].plot(hr_ref, x_ref, lw = 0.1, color = 'black')
    axs[0].plot(cr_alpha1, x_ref[:n_top+2], lw = 0.1, color = 'blue')
    axs[0].plot(cr_new_alpha, x_ref[:n_top+2], lw = 0.1, color = 'orange')

    axs[1].plot(cr_alpha1-hr_ref[:n_top+2], x_ref[:n_top+2], lw = 0.1, color = 'blue')
    axs[1].plot(cr_new_alpha-hr_ref[:n_top+2], x_ref[:n_top+2], lw = 0.1, color = 'orange')

fig.savefig(cart_out + 'global_check_spaghetti{}.pdf'.format(version))


fit_new = np.stack(fit_new)
fit_alpha1 = np.stack(fit_alpha1)

pickle.dump([fit_alpha1, fit_new], open(cart_out + 'hr_check_refit_alpha_mipas.p', 'wb'))

x_fomi = np.arange(2., 17.26, 0.25)

# for ii in range(len(obs)):
#     spl = spline(mipx[ii], -obs[ii], extrapolate = False)
#     hr_ref = spl(x_ref)
#     cr_alpha1 = fit_alpha1[ii]
#     cr_new_alpha = fit_new[ii]
#
#     axs[0].plot(hr_ref, x_ref, lw = 0.1, color = 'black')
#     axs[0].plot(cr_alpha1, x_ref[:n_top+2], lw = 0.1, color = 'blue')
#     axs[0].plot(cr_new_alpha, x_ref[:n_top+2], lw = 0.1, color = 'orange')
#
#     axs[1].plot(cr_alpha1-hr_ref[:n_top+2], x_ref[:n_top+2], lw = 0.1, color = 'blue')
#     axs[1].plot(cr_new_alpha-hr_ref[:n_top+2], x_ref[:n_top+2], lw = 0.1, color = 'orange')

cr_fomi = []
cr_falp = []
cr_np = []
obsref = []
for ii in range(len(obs)):
    spl = spline(mipx[ii], -obs[ii], extrapolate = False)
    hr_ref = spl(x_ref)
    obsref.append(hr_ref)

    spl = spline(mipx[ii], new_param_fa[ii], extrapolate = False)
    cr_falp.append(spl(x_ref))

    spl = spline(mipx[ii], new_param[ii], extrapolate = False)
    cr_np.append(spl(x_ref))

    spl = spline(x_fomi, old_param[ii], extrapolate = False)
    crfo2 = spl(x_ref)
    cr_fomi.append(crfo2)

cr_fomi = np.stack(cr_fomi)
cr_falp = np.stack(cr_falp)
cr_np = np.stack(cr_np)
obsref = np.stack(obsref)

##################################################


d_fom = cr_fomi[:, :n_top+2] - obsref[:, :n_top+2]
d_new = cr_np[:, :n_top+2] - obsref[:, :n_top+2]
d_new_fa = cr_falp[:, :n_top+2] - obsref[:, :n_top+2]
d_refit = fit_new[:, :n_top+2] - obsref[:, :n_top+2]

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

dref_median = np.median(d_refit, axis = 0)
dref_1st = np.percentile(d_refit, 25, axis = 0)
dref_3rd = np.percentile(d_refit, 75, axis = 0)
dref_std = np.std(d_refit, axis = 0)

# ax.fill_betweenx(x_ref, dfo_mean-dfo_std, dfo_mean+dfo_std, color = 'blue', alpha = 0.4)
# ax.fill_betweenx(x_ref, dnw_mean-dnw_std, dnw_mean+dnw_std, color = 'red', alpha = 0.4)
# ax.plot(dfo_mean, x_ref, color = 'blue', lw = 2)
# ax.plot(dnw_mean, x_ref, color = 'red', lw = 2)

ax.fill_betweenx(x_ref[:n_top+2], dfo_1st, dfo_3rd, color = 'blue', alpha = 0.4)
ax.fill_betweenx(x_ref[:n_top+2], dnw_1st_fa, dnw_3rd_fa, color = 'orange', alpha = 0.4)
ax.fill_betweenx(x_ref[:n_top+2], dref_1st, dref_3rd, color = 'violet', alpha = 0.4)
ax.plot(dfo_median, x_ref[:n_top+2], color = 'blue', lw = 2, label = 'fomi')
ax.plot(dnw_median_fa, x_ref[:n_top+2], color = 'orange', lw = 2, label = 'fomialpha')
ax.plot(dref_median, x_ref[:n_top+2], color = 'violet', lw = 2, label = 'refit')

ax.grid()
ax.set_xlim(-10., 15.)
#ax.set_ylim(40., 110.)
ax.set_ylim(10., 18.)

ax.legend()

fig.savefig(cart_out + 'global_check_shading{}.pdf'.format(version))


#########################################
fig, axs = plt.subplots(3, 3, figsize = (16, 12))

lats = np.arange(-90, 91, 20)
for ax, lat1, lat2 in zip(axs.flatten(), lats[:-1], lats[1:]):
    cond = (gigi.latitude > lat1) & (gigi.latitude <= lat2)

    dfok = d_fom[cond]
    dnok = d_new[cond]
    dnok_fa = d_new_fa[cond]
    drefok = d_refit[cond]

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

    dref_median = np.median(drefok, axis = 0)
    dref_1st = np.percentile(drefok, 25, axis = 0)
    dref_3rd = np.percentile(drefok, 75, axis = 0)
    dref_std = np.std(drefok, axis = 0)

    ax.fill_betweenx(x_ref[:n_top+2], dfo_1st, dfo_3rd, color = 'blue', alpha = 0.4)
    ax.fill_betweenx(x_ref[:n_top+2], dnw_1st_fa, dnw_3rd_fa, color = 'orange', alpha = 0.4)
    ax.fill_betweenx(x_ref[:n_top+2], dref_1st, dref_3rd, color = 'violet', alpha = 0.4)
    ax.plot(dfo_median, x_ref[:n_top+2], color = 'blue', lw = 2)
    ax.plot(dnw_median_fa, x_ref[:n_top+2], color = 'orange', lw = 2)
    ax.plot(dref_median, x_ref[:n_top+2], color = 'violet', lw = 2)

    ax.grid()
    ax.set_xlim(-10., 10.)
    if lat2 == 90:
        ax.set_xlim(-15., 25.)
    #ax.set_ylim(40., 110.)
    ax.set_ylim(10., 18.)

    ax.set_title('{} to {}'.format(int(lat1), int(lat2)))


fig.savefig(cart_out + 'global_check_shading_latbands{}.pdf'.format(version))



# alpha_unif, alpha_dic_atm = pickle.load(open(cart_out_rep + 'alpha_singleatm.p', 'rb'))
#
# #### OK, and now.... regression model! with pc0, pc1 e tempgrad
# tempgrad = np.stack([np.gradient(atm_pt[(atm, 'temp')])[alt2:n_top+1] for atm in allatms])
# temps = np.stack([atm_pt[(atm, 'temp')][alt2:n_top+1] for atm in allatms])
#
#
# kbc = 0.69503
# E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition
# phifunz = np.exp(-E_fun/(kbc*temps))
#
# allcorrs = dict()
# eofs_all = dict()
#
# moddict = dict()
#
# popup = np.array([phifunz[ii]*cose_upper_atm[(atm, cco2, 'lamb')][alt2:n_top+1]/1.5988 for ii, atm in enumerate(allatms) for cco2 in range(1,8)])
#
# popup_mean = np.mean(popup, axis = 0)
# popup_anom = popup-popup_mean
#
# solver_pop = Eof(popup_anom)
#
# fig = plt.figure()
# for i in range(4):
#     plt.plot(solver_pop.eofs()[i], np.arange(alt2, n_top+1), label = str(i))
# plt.title('popup eofs')
# fig.savefig(cart_out_F + 'popup_eofs_allco2.pdf')
#
# figs3 = []
# alpha_fit = dict()
# alpha_fit_nl0 = dict()
# alpha_fit2 = dict()
# for cco2 in range(1,8):
#     alphaok = alpha_dic_atm[cco2]
#     alpha_min = np.min(alphaok, axis = 0)
#     alpha_max = np.max(alphaok, axis = 0)
#
#     lambdivA = np.array([cose_upper_atm[(atm, cco2, 'lamb')][alt2:n_top+1]/1.5988 for atm in allatms])
#     popup = lambdivA*phifunz
#
#     Xprods = []
#     for ee in range(4):
#         dotprods = np.array([np.dot(pop-popup_mean, solver_pop.eofs(eofscaling=1)[ee]) for pop in popup])
#         Xprods.append(dotprods)
#
#     X = np.stack(Xprods).T
#     #X = np.stack([pop_x0, pop_x1, pop_x2, pop_x3]).T
#
#     weights = np.array([1, 1, 1, 1, 1, 1])
#
#     ### model 1: multi-linear regression of first 4 popup pcs
#     scores = []
#     ints = []
#     coefs = []
#     for ii in range(n_trans):
#         Y = np.stack(alphaok[:,ii])
#         model1 = LinearRegression().fit(X, Y, weights)
#         print(ii, model1.score(X, Y))
#         scores.append(model1.score(X, Y))
#         ints.append(model1.intercept_)
#         coefs.append(model1.coef_)
#         print('\n')
#
#     alpha_fit[cco2] = np.concatenate([np.array(ints)[:, np.newaxis], np.stack(coefs)], axis = 1)
#
#     ### model 2: parabolic fit to 1st pc, multi-linear regression of residuals with other 2 popup pcs
#     Xprods = []
#     for ee in range(2):#3):
#         dotprods = np.array([np.dot(pop-popup_mean, solver_pop.eofs(eofscaling=1)[ee]) for pop in popup])
#         Xprods.append(dotprods)
#         #if ee == 0:
#         Xprods.append(dotprods**2)
#
#     X2 = np.stack(Xprods).T
#
#     scores = []
#     ints = []
#     coefs = []
#     for ii in range(n_trans):
#         Y = np.stack(alphaok[:,ii])
#         model1 = LinearRegression().fit(X2, Y, weights)
#         print(ii, model1.score(X2, Y))
#         scores.append(model1.score(X2, Y))
#         ints.append(model1.intercept_)
#         coefs.append(model1.coef_)
#         print('\n')
#
#     alpha_fit2[cco2] = np.concatenate([np.array(ints)[:, np.newaxis], np.stack(coefs)], axis = 1)
#
#     fig3, ax3 = plt.subplots()
#     for ii, (atm, col) in enumerate(zip(allatms, npl.color_set(6))):
#         ls = '--'
#         alpha = alpha_fit[cco2][:, 0] + np.sum(alpha_fit[cco2][:, 1:] * X[ii][np.newaxis, :], axis = 1)
#
#         alpha[alpha < alpha_min] = alpha_min[alpha < alpha_min]
#         alpha[alpha > alpha_max] = alpha_max[alpha > alpha_max]
#
#         ax3.plot((alpha-alphaok[ii, :])/alphaok[ii, :], np.arange(alt2, n_top+1), color = col, linestyle = ls, label = atm)
#
#         # ax3.plot((alpha_unif[cco2-1]-alphaok[ii, :])/alphaok[ii, :], np.arange(alt2, n_top+1), color = col, linestyle = ':')
#
#     # plt.legend()
#     # plt.title('cco2: ' + str(cco2) + '- mod lin')
#     # plt.xlabel('relative error in alpha')
#     # figs3.append(fig3)
#
#     #fig3, ax3 = plt.subplots()
#     for ii, (atm, col) in enumerate(zip(allatms, npl.color_set(6))):
#         ls = ':'
#         alpha = alpha_fit2[cco2][:, 0] + np.sum(alpha_fit2[cco2][:, 1:] * X2[ii][np.newaxis, :], axis = 1)
#         alpha[alpha < alpha_min] = alpha_min[alpha < alpha_min]
#         alpha[alpha > alpha_max] = alpha_max[alpha > alpha_max]
#         #alpha = np.array([np.polyval(alpha_fit_nl0[cco2][izz, :], X[ii, 0]) for izz in range(len(alpha_fit_nl0[cco2]))]) + alpha_fit2[cco2][:, 0] +  np.sum(alpha_fit2[cco2][:, 1:] * X[ii][np.newaxis, 1:-1], axis = 1)
#         ax3.plot((alpha-alphaok[ii, :])/alphaok[ii, :], np.arange(alt2, n_top+1), color = col, linestyle = ls, label = atm)
#
#         # ax3.plot((alpha_unif[cco2-1]-alphaok[ii, :])/alphaok[ii, :], np.arange(alt2, n_top+1), color = col, linestyle = ':')
#
#     plt.legend()
#     plt.title('cco2: ' + str(cco2) + '- lin vs non lin')
#     plt.xlabel('relative error in alpha')
#     figs3.append(fig3)
#
#
# npl.plot_pdfpages(cart_out_rep + 'check_alpha_popup_relerr_v8.pdf', figs3)
#
# #sys.exit()
# #####################################################################
#
# ####################################################################
#
# alpha_fit['popup_mean'] = popup_mean
# alpha_fit['eof0'] = solver_pop.eofs(eofscaling=1)[0]
# alpha_fit['eof1'] = solver_pop.eofs(eofscaling=1)[1]
# alpha_fit['eof2'] = solver_pop.eofs(eofscaling=1)[2]
# alpha_fit['eof3'] = solver_pop.eofs(eofscaling=1)[3]
#
# for cco2 in range(1, 8):
#     alpha_fit[('min', cco2)] = np.min(alpha_dic_atm[cco2], axis = 0)
#     alpha_fit[('max', cco2)] = np.max(alpha_dic_atm[cco2], axis = 0)
#
# pickle.dump(alpha_fit, open(cart_out_rep + 'alpha_fit_4e.p', 'wb'))
#
# alpha_fit2['popup_mean'] = popup_mean
# alpha_fit2['eof0'] = solver_pop.eofs(eofscaling=1)[0]
# alpha_fit2['eof1'] = solver_pop.eofs(eofscaling=1)[1]
#
# for cco2 in range(1, 8):
#     alpha_fit2[('min', cco2)] = np.min(alpha_dic_atm[cco2], axis = 0)
#     alpha_fit2[('max', cco2)] = np.max(alpha_dic_atm[cco2], axis = 0)
# pickle.dump(alpha_fit2, open(cart_out_rep + 'alpha_fit_nl0.p', 'wb'))
#
#
# ############################################################
# tot_coeff_co2 = pickle.load(open(cart_out_2 + 'tot_coeffs_co2_NLTE.p', 'rb'))
#
# figs = []
# a0s = []
# a1s = []
# for cco2 in range(1, 8):
#     alpha_min = np.min(alpha_dic_atm[cco2], axis = 0)
#     alpha_max = np.max(alpha_dic_atm[cco2], axis = 0)
#
#     for ii, atm in enumerate(allatms):
#         temp = atm_pt[(atm, 'temp')]
#         surf_temp = atm_pt[(atm, 'surf_temp')]
#         pres = atm_pt[(atm, 'pres')]
#
#         co2vmr = atm_pt[(atm, cco2, 'co2')]
#         ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
#         o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
#         n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]
#
#         L_esc = cose_upper_atm[(atm, cco2, 'L_esc_all_wutop')]
#         lamb = cose_upper_atm[(atm, cco2, 'lamb')]
#         MM = cose_upper_atm[(atm, cco2, 'MM')]
#
#         hr_calc = npl.hr_reparam_low(cco2, temp, surf_temp, regrcoef = regrcoef, nlte_corr = nlte_corr)
#
#         hra, hrb = npl.hr_from_ab_diagnondiag(all_coeffs[(atm, cco2, 'acoeff')], all_coeffs[(atm, cco2, 'bcoeff')], all_coeffs[(atm, cco2, 'asurf')], all_coeffs[(atm, cco2, 'bsurf')], atm_pt[(atm, 'temp')], atm_pt[(atm, 'surf_temp')], max_alts = 66)
#         hra = hra[alt2:n_top+1]
#         hrb = hrb[alt2:n_top+1]
#
#         # population upper state
#         phifunz = np.exp(-E_fun/(kbc*temp[alt2:n_top+1]))
#         lambdivA = lamb[alt2:n_top+1]/1.5988
#         popup = lambdivA*phifunz
#
#         dotprods = np.array([np.dot(popup-popup_mean, eoff) for eoff in solver_pop.eofs(eofscaling=1)[:4]])
#
#         alpha6 = alpha_fit[cco2][:, 0] + np.sum(alpha_fit[cco2][:, 1:] * dotprods[np.newaxis, :], axis = 1)
#
#         print('setting constraint on alpha! check this part')
#         alpha6[alpha6 < alpha_min] = alpha_min[alpha6 < alpha_min]
#         alpha6[alpha6 > alpha_max] = alpha_max[alpha6 > alpha_max]
#
#         if cco2 == 7:
#             print('AAAAA')
#             print(alpha6)
#
#         alphaok = alpha_dic_atm[cco2][ii, :]
#         if cco2 == 7: print(alphaok)
#
#         #dotprods2 = np.array([dotprods[0], dotprods[0]**2] + list(dotprods[1:-1]))
#         dotprods2 = np.array([dotprods[0], dotprods[0]**2] + [dotprods[1], dotprods[1]**2])
#         alpha_nl = alpha_fit2[cco2][:, 0] + np.sum(alpha_fit2[cco2][:, 1:] * dotprods2[np.newaxis, :], axis = 1)
#
#         print('setting constraint on alpha! check this part')
#         alpha_nl[alpha_nl < alpha_min] = alpha_min[alpha_nl < alpha_min]
#         alpha_nl[alpha_nl > alpha_max] = alpha_max[alpha_nl > alpha_max]
#         # alpha_nl = np.array([np.polyval(alpha_fit_nl0[cco2][izz, :], dotprods[0]) for izz in range(len(alpha_fit_nl0[cco2]))]) + alpha_fit2[cco2][:, 0] +  np.sum(alpha_fit2[cco2][:, 1:] * dotprods[np.newaxis, 1:-1], axis = 1)
#
#         #n_top = alt2 + 10
#         hr_calc6 = npl.recformula(alpha6, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top)
#         hr_calc_nl = npl.recformula(alpha_nl, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top)
#
#         # hr_calc_aok = npl.recformula(alphaok, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top)
#         hr_calc_aunif = npl.recformula(alpha_unif[cco2-1, :], L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top)
#
#         hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_ref')]
#         hr_calc_vf5 = npl.new_param_full_old(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr)#, coeffs = coeffs_NLTE)
#
#         # alt_fomi, hr_fomi = npl.old_param(all_alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr)
#         # oldco = spline(alt_fomi, hr_fomi)
#         # hr_fomi = oldco(alts)
#
#         tit = 'co2: {} - atm: {}'.format(cco2, atm)
#         xlab = 'CR (K/day)'
#         ylab = 'index'
#         # labels = ['nlte_ref'] + [modnams[4]] + [modnams[5]] + ['alphaok', 'vf5', 'auni']
#         # hrs = [hr_ref, hr_calc5, hr_calc6, hr_calc_aok, hr_calc_vf5, hr_calc_aunif]
#         labels = ['nlte_ref', 'pop_4e_wt', 'pop_nl0_wt', 'aunif', 'vf5']#, 'fomi']
#         hrs = [hr_ref, hr_calc6, hr_calc_nl, hr_calc_aunif, hr_calc_vf5]#, hr_fomi]
#
#         colors = npl.color_set(5)
#         fig, a0, a1 = npl.manuel_plot(np.arange(66), hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-10, 10), xlim = (-70, 10), ylim = (40, 67), linestyles = ['-', '--', ':', ':', ':', ':', ':'], colors = colors, orizlines = [40, alt2, n_top])
#
#         figs.append(fig)
#         a0s.append(a0)
#         a1s.append(a1)
#
#         npl.adjust_ax_scale(a0s)
#         npl.adjust_ax_scale(a1s)
#
# npl.plot_pdfpages(cart_out_F + 'check_reparam_high_v8.pdf', figs)
