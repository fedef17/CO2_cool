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

co2profs = [atm_pt[('mle', cco2, 'co2')] for cco2 in range(1,8)]

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


cose_upper_atm = pickle.load(open(cart_out_3 + 'cose_upper_atm.p', 'rb'))
alpha_dic = pickle.load(open(cart_out_3 + 'alpha_upper.p', 'rb'))
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

n_top = n_alts_trhi+5
alpha = alpha_dic[(n_top, 'L_esc_all_wutop', 'least_squares', cco2)]

####################################################################################

print('Coeffs from interpolation!')
calc_coeffs = dict()
for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
    int_fun = interp_coeffs[(nam, 'int_fun')]
    sc = interp_coeffs[(nam, 'signc')]

    coeff = npl.coeff_from_interp(int_fun, sc, co2vmr)
    calc_coeffs[nam] = coeff

    if np.max(np.abs((coeff - coeffs_NLTE[nam][cco2-1, ...])/coeff)) > 1.e-10:
        print('AAAAAAAAAAAAAAAAAAAAAAARGH', nam)
        print(coeff, coeffs_NLTE[nam][cco2-1, ...])

hr_calc = npl.hr_from_ab(calc_coeffs['acoeff'], calc_coeffs['bcoeff'], calc_coeffs['asurf'], calc_coeffs['bsurf'], temp, surf_temp)

#alpha_ = 10.*np.ones(n_alts_trhi-n_alts_trlo+1)
hr_calc = npl.recformula(alpha, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = n_alts_trlo, n_alts_trhi = n_top)

hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_nlte')]
hr_ref[:n_alts_lte] = all_coeffs_nlte[(atm, cco2, 'hr_lte')][:n_alts_lte]

alt_fomi, hr_fomi = npl.old_param(alts, temp, pres, co2vmr)
oldco = spline(alt_fomi, hr_fomi)
hr_fomi = oldco(alts)

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
#zcool = npl.new_param_full(temp, pres, CO2prof, coeffs = coeffs_NLTE) # Add O, O2, N2 profile?

# Check per ogni atm

# Check con atm nuova?
