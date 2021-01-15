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

sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'
cart_out_2 = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE/'

cart_out_3 = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE_upper/'
if not os.path.exists(cart_out_3): os.mkdir(cart_out_3)

import newparam_lib as npl
import spect_base_module as sbm
#from scipy.interpolate import UnivariateSpline as spline

"""
This is for the recurrence formula:
3) Región x=12.5-13.75, formula de recurrencia, Eq. 7, con la función escape (L) corregida con el factor alpha (Eq. 12) con objeto de incluir el intercambio radiativo con las capas de arriba y las contribuciones de las bandas débiles (todas menos la fundamental).
4) Región x=14.0-16.5, formula de recurrencia, Eq. 7, con la función escape (L) pero sin corregir con alpha. De esa forma, al no corregir, se entiende que el intercambio radiativo con las capas de arriba y las contribuciones de las bandas débiles (todas menos la fundamental) son despreciables.
"""
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

# OK. now.
# Prendo gli epsilon alla top quota di sotto: 54. mi ricavo epsilon gn. estendo in alto con la recurrence formula. Poi aggiungo il fattore alpha.
# si parte dai d_j. Faccio separatamente per ogni co2prof.

atm = 'mle'
cco2 = 3

L_esc = all_coeffs_nlte[(atm, cco2, 'l_esc')]
uco2 = all_coeffs_nlte[(atm, cco2, 'uco2')]
Lspl = spline(uco2, L_esc, extrapolate = False)

pres = atm_pt[(atm, 'pres')]
temp = atm_pt[(atm, 'temp')]
surf_temp = atm_pt[(atm, 'surf_temp')]
n_dens = sbm.num_density(pres, temp)
co2vmr = atm_pt[(atm, cco2, 'co2')]
n_co2 = n_dens * co2vmr

phi_fun = np.exp(-E_fun/(kbc*temp))

ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]
# o2vmr = np.ones(len(alts))*0.20-ovmr
# n2vmr = np.ones(len(alts))*0.79

###################### Rate coefficients ######################
t13 = temp**(-1./3)

# Collisional rate between CO2 and O:
zo = 3.5e-13*np.sqrt(temp)+2.32e-9*np.exp(-76.75*t13) # use Granada parametrization
#ZCO2O = KO Fomichev value

# Collisional rates between CO2 and N2/O2:
zn2=7e-17*np.sqrt(temp)+6.7e-10*np.exp(-83.8*t13)
zo2=7e-17*np.sqrt(temp)+1.0e-9*np.exp(-83.8*t13)
# use Fomichev values
# zn2=5.5e-17*sqrt(temp)+6.7e-10*exp(-83.8*t13)
# zo2=1.e-15*exp(23.37-230.9*t13+564.*t13*t13)

###############################################################

uok = []
for ial in range(len(alts)):
    uok.append(np.trapz(n_co2[ial:], 1.e5*alts[ial:])) # integro in cm, voglio la colonna in cm-2

uok = np.array(uok)
Lok = Lspl(uok)
Lok[:20][np.isnan(Lok[:20])] = 0.0 # for extrapolated regions
Lok[-5:][np.isnan(Lok[-5:])] = 1.0 # for extrapolated regions

alpha = np.ones(len(Lok)) # depends on cco2
eps_gn = np.zeros(len(Lok))

alpha[n_alts_trlo-1:n_alts_trhi] = np.array([1.8, 1.7, 1.5, 1.3, 1.1, 1.05, 1.0])

dj = alpha*Lok
lamb = 1.5988/(1.5988 + n_dens*(n2vmr*zn2 + o2vmr*zo2 + ovmr*zo))

## Boundary condition
#eps125 = all_coeffs_nlte[(atm, cco2, 'hr_nlte')][n_alts_trlo]
tip = 'varfit5_nlte'
acoeff_cco2 = tot_coeff_co2[(tip, 'acoeff', cco2)]
bcoeff_cco2 = tot_coeff_co2[(tip, 'bcoeff', cco2)]
asurf_cco2 = tot_coeff_co2[(tip, 'asurf', cco2)]
bsurf_cco2 = tot_coeff_co2[(tip, 'bsurf', cco2)]

hr_calc = npl.hr_from_ab(acoeff_cco2, bcoeff_cco2, asurf_cco2, bsurf_cco2, temp, surf_temp)
eps125 = hr_calc[n_alts_trlo-1]

hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_nlte')]
eps14 = hr_ref[n_alts_trhi-1]

##### convert from K day-1 to erg g-1 s-1
eps125 = eps125 * cp / (24*60*60)
eps14 = eps14 * cp / (24*60*60)

#eps_gn[n_alts_trlo-1] = 1.10036e-10*eps125/(co2vmr[n_alts_trlo-1] * (1-lamb[n_alts_trlo-1]))
eps_gn[n_alts_trhi-1] = 1.10036e-10*eps14/(co2vmr[n_alts_trhi-1] * (1-lamb[n_alts_trhi-1]))

coso = '{:4d}  {:7.1e}  {:7.1e}  {:7.1e}  {:7.1e}    {:7.1e}  {:7.1e}  {:7.1e}'
#for j in range(n_alts_trlo, n_alts): # Formula 9
for j in range(n_alts_trhi, n_alts): # Formula 9
    Djj = 0.25*(dj[j-1] + 3*dj[j])
    Djjm1 = 0.25*(dj[j] + 3*dj[j-1])

    Fj = (1 - lamb[j]*(1-Djj))
    Fjm1 = (1 - lamb[j-1]*(1-Djjm1))
    print(coso.format(j, Fjm1*eps_gn[j-1], Djjm1*phi_fun[j-1], Djj*phi_fun[j], Fj, Fjm1, Djjm1, Djj))
    eps_gn[j] = (Fjm1*eps_gn[j-1] + Djjm1*phi_fun[j-1] - Djj*phi_fun[j])/Fj

MM = np.ones(len(alts)) * (0.79*28+0.21*32) # Molecular mass
fac = (2.63187e11 * co2vmr * (1-lamb))/MM
eps = fac * eps_gn # Formula 7

eps = eps * (24*60*60) / cp # convert back to K/day

###### IMPORTANT!! UNCOMMENT FOR COOL-TO-SPACE region
# now for the cs region:
# Phi_165 = eps_gn[n_alts_cs] + phi_fun[n_alts_cs]
# eps[n_alts_cs:] = fac[n_alts_cs:] * (Phi_165 - phi_fun[j])

### Putting all in hr_calc
#hr_calc[n_alts_trlo:] = eps[n_alts_trlo:]
hr_calc[n_alts_trhi:] = eps[n_alts_trhi:]
hr_calc[:n_alts_trhi] = hr_ref[:n_alts_trhi]
hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_nlte')]
hr_ref[:n_alts_lte] = all_coeffs_nlte[(atm, cco2, 'hr_lte')][:n_alts_lte]

alt_fomi, hr_fomi = npl.old_param(alts, temp, pres, co2vmr)
oldco = spline(alt_fomi, hr_fomi)
hr_fomi = oldco(alts)

tit = 'co2: {} - atm: {}'.format(cco2, atm)
xlab = 'CR (K/day)'
ylab = 'Alt (km)'
labels = ['nlte_ref', 'new_param', 'old param']
hrs = [hr_ref, hr_calc, hr_fomi]
#labels = ['ref'] + alltips + ['fomi rescale (no fit)', 'old param']
fig, a0, a1 = npl.manuel_plot(alts, hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-8, 8), xlim = (-40, 10), linestyles = ['-', '-', ':'], orizlines = [70., alts[n_alts_trlo], alts[n_alts_trhi]])

fig.savefig(cart_out_3 + 'check_upper_NOalpha_mle_3_samestart.pdf')

# figs.append(fig)
# a0s.append(a0)
# a1s.append(a1)
#
# npl.adjust_ax_scale(a0s)
# npl.adjust_ax_scale(a1s)
#
# npl.plot_pdfpages(cart_out_2 + 'check_newparam_NLTE_lowtrans.pdf', figs)
