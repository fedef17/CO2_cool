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
from scipy.interpolate import interp1d

if os.uname()[1] == 'xaru':
    cart_base = '/home/fedef/Research/'
elif os.uname()[1] == 'hobbes':
    cart_base = '/home/fabiano/Research/'

sys.path.insert(0, cart_base + 'git/SpectRobot/')
sys.path.insert(0, cart_base + 'git/pythall/')

cart_co2 = cart_base + 'lavori/CO2_cooling/new_param/'
cart_out = cart_co2 + 'LTE/'
cart_out_2 = cart_co2 + 'NLTE/'
cart_out_3 = cart_co2 + 'NLTE_upper/'
cart_out_rep = cart_co2 + 'NLTE_reparam/'

cart_out_4 = cart_co2 + 'reparam_allatm/'
if not os.path.exists(cart_out_4): os.mkdir(cart_out_4)

import newparam_lib as npl
import spect_base_module as sbm
from scipy.optimize import Bounds, minimize, least_squares

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

alts = atm_pt[('mle', 'alts')]

pres = atm_pt[('mle', 'pres')]
x = np.log(1000./pres)
n_alts_trlo = np.sum(x < 12.5)
print(n_alts_trlo)
#print('low trans at {}'.format(alts[n_alts_trlo]))

all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))
cose_upper_atm = pickle.load(open(cart_out_3 + 'cose_upper_atm.p', 'rb'))
#alpha_dic = pickle.load(open(cart_out_3 + 'alpha_upper.p', 'rb'))

################################################################
### Loading new coefficients

regrcoef = pickle.load(open(cart_out_rep + 'regrcoef_v3.p', 'rb')) #### for LTE region

nlte_corr = pickle.load(open(cart_out_rep + 'nlte_corr_low.p', 'rb')) #### NLTE correction for low trans region

#### Fit of alpha in upper transition region
alpha_fit_4e = pickle.load(open(cart_out_rep + 'alpha_fit_4e.p', 'rb'))
alpha_fit_nl0 = pickle.load(open(cart_out_rep + 'alpha_fit_nl0.p', 'rb'))

################################################################################

pres = atm_pt[('mle', 'pres')]
x = np.log(1000./pres)
n_alts_trlo = np.sum(x < 12.5)
print('low trans at {}, {:7.2f} km, {:7.2f}'.format(n_alts_trlo, alts[n_alts_trlo], x[n_alts_trlo]))

n_alts_trhi = np.sum(x < 14)
print('high trans at {}, {:7.2f} km {:7.2f}'.format(n_alts_trhi, alts[n_alts_trhi], x[n_alts_trhi]))

# il cool-to-space è fuori dalle 66 alts
# n_alts_cs = np.sum(x < 16.5)
# print('cool-to-space at {}, {:7.2f} km'.format(n_alts_cs, alts[n_alts_cs]))

n_alts_lte = 40

#############################################################
# Crea tabelle coeffs e salvale in pickle separati. (da convertire poi in file di testo o netcdf per Bernd)
# che coeffs mi servono:
#   - i regrcoef della parte LTE: regrcoef[(cco2, conam, 'c1')], m1, m2
#   - i coeffs della nlte_correction
#   - gli L
#   - i coeffs per il fit degli alpha
# per ogni a,b,ecc coeff faccio una matrice con prima dimensione quella della co2

#####################################################
alt1 = 40
alt2 = 51
n_top = 65

co2profs = np.stack([atm_pt[('mle', cco2, 'co2')] for cco2 in range(1,8)])

#### LTE part
coeffs_fin = dict() ## le matricione
interp_coeffs = dict() ## l'interpolazione
for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
    kosi = ['c1', 'm1', 'm2']
    if 'surf' in nam:
        kosi = ['c', 'm']

    for regco in kosi:
        ko = np.stack([regrcoef[(cco2, nam, regco)] for cco2 in range(1, 8)])
        coeffs_fin[(nam, regco)] = ko
        # BERND. QUI SE DEVI SCRIVERLO COME FILE TXT o netcdf

        if regco in ['c', 'c1']:
            ## this only works for the constant term. The others (m, m1, m2) change sign in some points and can be "logged"
            int_fun, signc = npl.interp_coeff_logco2(ko, co2profs)
            interp_coeffs[(nam, regco, 'int_fun')] = int_fun
            interp_coeffs[(nam, regco, 'signc')] = signc
        else:
            int_fun = npl.interp_coeff_linco2(ko, co2profs)
            interp_coeffs[(nam, regco, 'int_fun')] = int_fun

## BERND. Other things to be saved for LTE: regrcoef['surfmean'], regrcoef['amean'], regrcoef['eof0'], regrcoef['eof1']

#### NLTE correction (low trans)
nam = 'nltecorr'
for regco in ['c', 'm1', 'm2', 'm3', 'm4']:
    ko = np.stack([nlte_corr[(cco2, regco)] for cco2 in range(1, 8)])
    coeffs_fin[(nam, regco)] = ko
    # BERND. QUI SE DEVI SCRIVERLO COME FILE TXT o netcdf

    int_fun = npl.interp_coeff_linco2(ko, co2profs[:, alt1:alt2])
    interp_coeffs[(nam, regco, 'int_fun')] = int_fun
    #interp_coeffs[(nam, regco, 'signc')] = signc

##########################################################

#BERND. remember to add here: alpha_fit['popup_mean'], alpha_fit['eof0'], ... alpha_fit['eof3']

alphas_all = np.stack([alpha_fit_nl0[cco2] for cco2 in range(1,8)])
coeffs_fin['alpha'] = alphas_all
intfutu = []
for go in range(alphas_all.shape[-1]):
    int_fun = npl.interp_coeff_linco2(alphas_all[..., go], co2profs[:, alt2:n_top+1])
    intfutu.append(int_fun)
interp_coeffs[('alpha', 'int_fun')] = intfutu
#interp_coeffs[('alpha', 'signc')] = signc

coeffs_fin['alpha_min'] = np.stack([alpha_fit_nl0[('min', i)] for i in range(1,8)])
interp_coeffs[('alpha_min', 'int_fun')] = npl.interp_coeff_linco2(coeffs_fin['alpha_min'], co2profs[:, alt2:n_top+1])

coeffs_fin['alpha_max'] = np.stack([alpha_fit_nl0[('max', i)] for i in range(1,8)])
interp_coeffs[('alpha_max', 'int_fun')] = npl.interp_coeff_linco2(coeffs_fin['alpha_max'], co2profs[:, alt2:n_top+1])

L_all = np.stack([np.mean([all_coeffs_nlte[(atm, cco2, 'l_esc')] for atm in allatms], axis = 0) for cco2 in range(1,8)])
uco2 = all_coeffs_nlte[('mle', 1, 'uco2')] # same for all
coeffs_fin['Lesc'] = L_all
coeffs_fin['uco2'] = uco2

# Lesc_all = np.stack([cose_upper_atm[('mle', cco2, 'L_esc_all_wutop')] for cco2 in range(1,8)])
# Lesc_all[np.isnan(Lesc_all)] = 0.
# coeffs_fin['Lesc'] = Lesc_all
int_fun = npl.interp_coeff_linco2(L_all, co2profs)
interp_coeffs[('Lesc', 'int_fun')] = int_fun
#interp_coeffs[('Lesc', 'signc')] = signc

coeffs_fin['co2profs'] = co2profs
pickle.dump(coeffs_fin, open(cart_out_4 + 'coeffs_finale.p', 'wb'))

#############################################################

####################################################################################
# Check per un atm
atm = 'mls'
cco2 = 6

pres = atm_pt[(atm, 'pres')]
temp = atm_pt[(atm, 'temp')]
surf_temp = atm_pt[(atm, 'surf_temp')]

L_esc = cose_upper_atm[(atm, cco2, 'L_esc_all_wutop')]

lamb = cose_upper_atm[(atm, cco2, 'lamb')] # Qui c'è la info degli altri gas e dei coefficienti
co2vmr = cose_upper_atm[(atm, cco2, 'co2vmr')]
MM = cose_upper_atm[(atm, cco2, 'MM')]

# alpha = alpha_dic[(n_top, 'L_esc_all_wutop', 'least_squares', cco2)]

print('Sample check -> coeffs from interpolation')
calc_coeffs = dict()
for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf', 'nltecorr']:
    for regco in ['c', 'm', 'c1', 'm1', 'm2', 'm3', 'm4']:
        if (nam, regco, 'int_fun') not in interp_coeffs:
            continue
        int_fun = interp_coeffs[(nam, regco, 'int_fun')]

        if regco in ['c', 'c1'] and nam != 'nltecorr':
            sc = interp_coeffs[(nam, regco, 'signc')]
            coeff = npl.coeff_from_interp_log(int_fun, sc, co2vmr)
        else:
            coeff = npl.coeff_from_interp_lin(int_fun, co2vmr)

        calc_coeffs[(nam, regco)] = coeff

        if np.max(np.abs((coeff - coeffs_fin[(nam, regco)][cco2-1, ...])/coeff)) > 1.e-10:
            print('AAAAAAAAAAAAAAAAAAAAAAARGH', nam)
            print(coeff, coeffs_fin[(nam, regco)][cco2-1, ...])

#lte
acoeff, bcoeff, asurf, bsurf = npl.coeffs_from_eofreg_single(temp, surf_temp, calc_coeffs)
hr_lte = npl.hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = 66)

#### nltecorr
hr_nlte_corr = npl.nltecorr_from_eofreg_single(temp, surf_temp, calc_coeffs, alt1 = alt1, alt2 = alt2)

hr_calc = hr_lte.copy()
hr_calc[alt1:alt2] += hr_nlte_corr

#### upper atm
intfutu = interp_coeffs[('alpha', 'int_fun')]
#sc = interp_coeffs[('alpha', 'signc')]
allco = []
for intfu in intfutu:
    allco.append(npl.coeff_from_interp_lin(intfu, co2vmr[alt2:n_top+1]))
calc_coeffs['alpha_fit'] = np.stack(allco).T

calc_coeffs['alpha_min'] = npl.coeff_from_interp_lin(interp_coeffs[('alpha_min', 'int_fun')], co2vmr[alt2:n_top+1])
calc_coeffs['alpha_max'] = npl.coeff_from_interp_lin(interp_coeffs[('alpha_max', 'int_fun')], co2vmr[alt2:n_top+1])

if np.max(np.abs((calc_coeffs['alpha_fit'] - coeffs_fin['alpha'][cco2-1, ...])/calc_coeffs['alpha_fit'])) > 1.e-10:
    print('AAAAAAAAAAAAAAAAAAAAAAARGH', 'alpha')
    print(calc_coeffs['alpha_fit'], coeffs_fin['alpha'][cco2-1, ...])

alpha = npl.alpha_from_fit(temp, surf_temp, lamb, calc_coeffs['alpha_fit'], alpha_min = calc_coeffs['alpha_min'], alpha_max = calc_coeffs['alpha_max'])

hr_calc_fin = npl.recformula(alpha, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top)

###################################################

hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_nlte')]
hr_ref[:n_alts_lte] = all_coeffs_nlte[(atm, cco2, 'hr_lte')][:n_alts_lte]

ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]
#alt_fomi, hr_fomi = npl.old_param(alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr)
#oldco = spline(alt_fomi, hr_fomi)
#hr_fomi = oldco(alts)

### FIGURE
tit = 'co2: {} - atm: {}'.format(cco2, atm)
xlab = 'CR (K/day)'
ylab = 'Alt (km)'
# labels = ['nlte_ref', 'new_param', 'np_wutop', 'np_all_wutop', 'np_aw_extended', 'np_noalpha', 'old param']
# hrs = [hr_ref, hr_calc, hr_calc_wutop, hr_calc_all, hr_calc_extended, hr_calc_alpha1, hr_fomi]
labels = ['nlte_ref', 'lte', 'nlte1', 'nlte2']
hrs = [hr_ref, hr_lte, hr_calc, hr_calc_fin]#, hr_fomi]

colors = np.array(npl.color_set(4))
fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-15, 15), xlim = (-70, 10), linestyles = ['-', '-', '--', ':'], colors = colors, orizlines = [70., alts[n_alts_trlo], alts[n_top]])

fig.savefig(cart_out_4 + 'test_calchr_{}_{}.pdf'.format(atm, cco2))

#################################################################

#ok.
all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

interp_coeffs_old = npl.precalc_interp_old()
interp_coeffs = npl.precalc_interp()

# Check per ogni atm
figs = []
a0s = []
a1s = []

for cco2 in range(1, 8):
    for atm in allatms:
        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]
        pres = atm_pt[(atm, 'pres')]

        co2vmr = atm_pt[(atm, cco2, 'co2')]
        ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
        o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
        n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]

        hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_nlte')]
        hr_ref[:n_alts_lte] = all_coeffs_nlte[(atm, cco2, 'hr_lte')][:n_alts_lte]

        hr_calc = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, alts, interp_coeffs = interp_coeffs)#, coeffs = coeffs_fin)
        hr_calc_old = npl.new_param_full_old(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs_old)

        alt_fomi, hr_fomi = npl.old_param(alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr)
        oldco = spline(alt_fomi, hr_fomi)
        hr_fomi = oldco(alts)

        tit = 'co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'Alt (km)'
        # labels = ['nlte_ref', 'new_param', 'np_wutop', 'np_all_wutop', 'np_aw_extended', 'np_noalpha', 'old param']
        # hrs = [hr_ref, hr_calc, hr_calc_wutop, hr_calc_all, hr_calc_extended, hr_calc_alpha1, hr_fomi]
        labels = ['nlte_ref', 'new param', 'param vf5', 'fomi']
        hrs = [hr_ref, hr_calc, hr_calc_old, hr_fomi]

        #colors = np.array(npl.color_set(3))
        colors = ['violet', 'forestgreen', 'indianred', 'steelblue']
        fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-15, 15), xlim = (-70, 10), ylim = (40, None), linestyles = ['-', '--', '--', '--'], colors = colors, orizlines = [70., alts[n_alts_trlo], alts[n_alts_trhi]])

        figs.append(fig)
        a0s.append(a0)
        a1s.append(a1)

        npl.adjust_ax_scale(a0s)
        npl.adjust_ax_scale(a1s)

npl.plot_pdfpages(cart_out_4 + 'check_allrefs_newparam.pdf', figs)


# Check con atm nuova?
figs = []
a0s = []
for atm in allatms:
    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]
    pres = atm_pt[(atm, 'pres')]

    ovmr = all_coeffs_nlte[(atm, 2, 'o_vmr')]
    o2vmr = all_coeffs_nlte[(atm, 2, 'o2_vmr')]
    n2vmr = all_coeffs_nlte[(atm, 2, 'n2_vmr')]

    new_cr = []
    old_cr = []
    mults = np.arange(0.25, 8.1, 0.25)
    for co2mult in mults:
        co2vmr = co2mult*atm_pt[(atm, 2, 'co2')]
        hr_calc = npl.new_param_full_old(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr)#, coeffs = coeffs_fin)
        new_cr.append(hr_calc)

        alt_fomi, hr_fomi = npl.old_param(alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr)
        oldco = spline(alt_fomi, hr_fomi)
        hr_fomi = oldco(alts)
        old_cr.append(hr_fomi)

    fig, ax = plt.subplots(figsize = (16, 12))
    colors = npl.color_set(len(mults))
    for nup, olp, col in zip(new_cr, old_cr, colors):
        ax.plot(nup, alts, color = col)
        ax.plot(olp, alts, color = col, linestyle = '--', linewidth = 0.5)

    ax.set_title('co2: 0.25-8.0 - atm: {}'.format(atm))
    ax.set_xlabel('CR (K/day)')
    ax.set_ylabel('Alt (km)')
    for orizli, col in zip([70., alts[n_alts_trlo], alts[n_alts_trhi]], ['red', 'orange', 'green']):
        ax.axhline(orizli, color = col, alpha = 0.6, linestyle = '--')
    ax.grid()

    figs.append(fig)
    a0s.append(ax)

npl.adjust_ax_scale(a0s)

npl.plot_pdfpages(cart_out_4 + 'rangeco2_newvsold.pdf', figs)

for ax in a0s:
    ax.set_ylim(40., 100.)
    ax.set_xlim(-20., 10.)

npl.plot_pdfpages(cart_out_4 + 'rangeco2_newvsold_zoom.pdf', figs)
