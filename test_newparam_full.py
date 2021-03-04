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

sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'
cart_out_2 = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE/'

cart_out_3 = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE_upper/'

cart_out_4 = '/home/fabiano/Research/lavori/CO2_cooling/new_param/newpar_allatm/'
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

n_alts = 66
alts = atm_pt[('mle', 'alts')]

pres = atm_pt[('mle', 'pres')]
x = np.log(1000./pres)
n_alts_trlo = np.sum(x < 12.5)
print('low trans at {}, {:7.2f} km, {:7.2f}'.format(n_alts_trlo, alts[n_alts_trlo], x[n_alts_trlo]))

n_alts_trhi = np.sum(x < 14)
print('high trans at {}, {:7.2f} km {:7.2f}'.format(n_alts_trhi, alts[n_alts_trhi], x[n_alts_trhi]))

# il cool-to-space è fuori dalle 66 alts
# n_alts_cs = np.sum(x < 16.5)
# print('cool-to-space at {}, {:7.2f} km'.format(n_alts_cs, alts[n_alts_cs]))

all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))
n_alts_lte = 40

#tot_coeff_co2 = pickle.load(open(cart_out + 'tot_coeffs_co2_v2_LTE.p', 'rb'))
tot_coeff_co2 = pickle.load(open(cart_out_2 + 'tot_coeffs_co2_NLTE.p', 'rb'))
cose_upper_atm = pickle.load(open(cart_out_3 + 'cose_upper_atm.p', 'rb'))
alpha_dic = pickle.load(open(cart_out_3 + 'alpha_upper.p', 'rb'))

n_top = n_alts_trhi+5

# Crea tabelle coeffs e salvale in pickle separati. (da convertire poi in file di testo o netcdf per Bernd)
# che coeffs mi servono:
#   - 'acoeff', 'bcoeff', 'asurf', 'bsurf' per tutti i co2. Questi sono quelli NLTE, che già contengono la modifica NLTE della low transition region
#   - gli L per tutti i co2
#   - gli alpha per la upper transition region, anche loro per tutti i co2

# per ogni a,b,ecc coeff faccio una matrice con prima dimensione quella della co2
tot_coeff_co2 = pickle.load(open(cart_out_2 + 'tot_coeffs_co2_NLTE.p', 'rb')) # qui ci sono sia i LTE che i NLTE
co2profs = np.stack([atm_pt[('mle', cco2, 'co2')] for cco2 in range(1,8)])

coeffs_NLTE = dict()
interp_coeffs = dict()
for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
    ko = np.stack([tot_coeff_co2[('varfit5_nlte', nam, cco2)] for cco2 in range(1, 8)])
    coeffs_NLTE[nam] = ko
    # QUI SE DEVI SCRIVERLO COME FILE TXT o netcdf

    int_fun, signc = npl.interp_coeff_logco2(ko, co2profs)
    interp_coeffs[(nam, 'int_fun')] = int_fun
    interp_coeffs[(nam, 'signc')] = signc

alphas_all = np.stack([alpha_dic[(n_top, 'L_esc_all_wutop', 'least_squares', cco2)] for cco2 in range(1,8)])
coeffs_NLTE['alpha'] = alphas_all
int_fun, signc = npl.interp_coeff_logco2(alphas_all, co2profs)
interp_coeffs[('alpha', 'int_fun')] = int_fun
interp_coeffs[('alpha', 'signc')] = signc

Lesc_all = np.stack([cose_upper_atm[('mle', cco2, 'L_esc_all_wutop')] for cco2 in range(1,8)])
coeffs_NLTE['Lesc'] = Lesc_all
int_fun, signc = npl.interp_coeff_logco2(Lesc_all, co2profs)
interp_coeffs[('Lesc', 'int_fun')] = int_fun
interp_coeffs[('Lesc', 'signc')] = signc

coeffs_NLTE['co2profs'] = co2profs
pickle.dump(coeffs_NLTE, open(cart_out_4 + 'coeffs_finale.p', 'wb'))

####################################################################################
# Check per un atm
atm = 'mle'
cco2 = 3

pres = atm_pt[(atm, 'pres')]
temp = atm_pt[(atm, 'temp')]
surf_temp = atm_pt[(atm, 'surf_temp')]

L_esc = cose_upper_atm[(atm, cco2, 'L_esc_all_wutop')]

lamb = cose_upper_atm[(atm, cco2, 'lamb')] # Qui c'è la info degli altri gas e dei coefficienti
co2vmr = cose_upper_atm[(atm, cco2, 'co2vmr')]
MM = cose_upper_atm[(atm, cco2, 'MM')]

alpha = alpha_dic[(n_top, 'L_esc_all_wutop', 'least_squares', cco2)]

####################################################################################

print('Coeffs from interpolation!')
calc_coeffs = dict()
for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf', 'alpha']:
    int_fun = interp_coeffs[(nam, 'int_fun')]
    sc = interp_coeffs[(nam, 'signc')]

    coeff = npl.coeff_from_interp(int_fun, sc, co2vmr)
    calc_coeffs[nam] = coeff

    if np.max(np.abs((coeff - coeffs_NLTE[nam][cco2-1, ...])/coeff)) > 1.e-10:
        print('AAAAAAAAAAAAAAAAAAAAAAARGH', nam)
        print(coeff, coeffs_NLTE[nam][cco2-1, ...])


hr_calc = npl.hr_from_ab(calc_coeffs['acoeff'], calc_coeffs['bcoeff'], calc_coeffs['asurf'], calc_coeffs['bsurf'], temp, surf_temp)

#alpha_ = 10.*np.ones(n_alts_trhi-n_alts_trlo+1)
hr_calc = npl.recformula(calc_coeffs['alpha'], L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = n_alts_trlo, n_alts_trhi = n_top)

hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_nlte')]
hr_ref[:n_alts_lte] = all_coeffs_nlte[(atm, cco2, 'hr_lte')][:n_alts_lte]

ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]
alt_fomi, hr_fomi = npl.old_param(alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr)
oldco = spline(alt_fomi, hr_fomi)
hr_fomi = oldco(alts)

### FIGURE
tit = 'co2: {} - atm: {}'.format(cco2, atm)
xlab = 'CR (K/day)'
ylab = 'Alt (km)'
# labels = ['nlte_ref', 'new_param', 'np_wutop', 'np_all_wutop', 'np_aw_extended', 'np_noalpha', 'old param']
# hrs = [hr_ref, hr_calc, hr_calc_wutop, hr_calc_all, hr_calc_extended, hr_calc_alpha1, hr_fomi]
labels = ['nlte_ref', 'new_param', 'old_param']
hrs = [hr_ref, hr_calc, hr_fomi]

colors = np.array(npl.color_set(3))
fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-15, 15), xlim = (-70, 10), linestyles = ['-', '-', '--'], colors = colors, orizlines = [70., alts[n_alts_trlo], alts[n_top]])

fig.savefig(cart_out_4 + 'test_calchr.pdf')

#ok.
all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

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

        hr_calc = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr)#, coeffs = coeffs_NLTE)

        alt_fomi, hr_fomi = npl.old_param(alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr)
        oldco = spline(alt_fomi, hr_fomi)
        hr_fomi = oldco(alts)

        tit = 'co2: {} - atm: {}'.format(cco2, atm)
        xlab = 'CR (K/day)'
        ylab = 'Alt (km)'
        # labels = ['nlte_ref', 'new_param', 'np_wutop', 'np_all_wutop', 'np_aw_extended', 'np_noalpha', 'old param']
        # hrs = [hr_ref, hr_calc, hr_calc_wutop, hr_calc_all, hr_calc_extended, hr_calc_alpha1, hr_fomi]
        labels = ['nlte_ref', 'new param', 'old param']
        hrs = [hr_ref, hr_calc, hr_fomi]

        colors = np.array(npl.color_set(3))
        colors = ['violet', 'steelblue', 'indianred']
        fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-15, 15), xlim = (-70, 10), linestyles = ['-', '-', '--'], colors = colors, orizlines = [70., alts[n_alts_trlo], alts[n_alts_trhi]])

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
        hr_calc = npl.new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr)#, coeffs = coeffs_NLTE)
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
