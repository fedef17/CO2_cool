#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
from matplotlib import cm
#import climtools_lib as ctl

from scipy import io
import scipy.constants as const
import pickle
from scipy.interpolate import PchipInterpolator as spline

if os.uname()[1] == 'ff-clevo':
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/SpectRobot/')
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/pythall/')
    cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'
else:
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'

import newparam_lib as npl

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

n_alts = 54

all_alts = atm_pt[('mle', 'alts')]
alts = atm_pt[('mle', 'alts')][:n_alts]

n_alts_lte = 40

tot_coeff_co2 = pickle.load(open(cart_out + 'tot_coeffs_co2_v2_LTE.p', 'rb'))

# Now. We are in the low transition region and need to adjust the LTE coeffs to non-LTE.
filsav = 'data_cira_{}_co2_{}.sav'
all_coeffs_nlte = dict()

for atm in allatms:
    for cco2 in range(1,8):
        coso = io.readsav(cartsav+filsav.format(atm, cco2))['data']
        nomi = 'HR_NLTE HR_NLTE_FB HR_NLTE_HOT HR_NLTE_ISO HR_LTE CO2_VMR O_VMR UCO2 L_ESC L_ESC_FOM'
        nomi = nomi.split()
        for nom in nomi:
            all_coeffs_nlte[(atm, cco2, nom.lower())] = getattr(coso, nom)[0]

# per ogni atm faccio:
for atm in allatms:
    for cco2 in range(1,8):
        hr_nlte = all_coeffs_nlte[(atm, cco2, 'hr_nlte')][:n_alts]
        hr_nlte_fun = all_coeffs_nlte[(atm, cco2, 'hr_nlte_fb')][:n_alts]+all_coeffs_nlte[(atm, cco2, 'hr_nlte_iso')][:n_alts]
        hr_nlte_hot = all_coeffs_nlte[(atm, cco2, 'hr_nlte_hot')][:n_alts]
        hr_lte = all_coeffs_nlte[(atm, cco2, 'hr_lte')][:n_alts]

        hr_calc
        for tip in ['unifit', 'varfit', 'varfit2', 'varfit3', 'varfit4', 'varfit5']:
        acoeff_cco2 = tot_coeff_co2[(tip, 'acoeff', cco2)]
        bcoeff_cco2 = tot_coeff_co2[(tip, 'bcoeff', cco2)]
        asurf_cco2 = tot_coeff_co2[(tip, 'asurf', cco2)]
        bsurf_cco2 = tot_coeff_co2[(tip, 'bsurf', cco2)]

            hr_calc = npl.hr_from_ab(acoeff_cco2, bcoeff_cco2, asurf_cco2, bsurf_cco2, temp, surf_temp)[:n_alts]
        for cnam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
            all_coeffs_nlte[(atm, cco2, cnam)] = all_coeffs[(atm, cco2, cnam)]*h_ref/h_lte
a_nlte = a * h_ref/h_lte
b_nlte = b * h_ref/h_lte

pickle.dump(open(cart_out + 'all_coeffs_NLTE.p', 'wb'))

# Poi devo ripetere la roba di fomi_multiatmco2_ab_LTE_v3 per a e b separatamente.
# Cioè. tengo b della singola atm, b_nlte, e fitto i weights delle diverse atms per a.
# con questi weights ottengo un a per tutte le atm. Tengo ora fisso questo a e cerco i weights per b. Quindi i weights di a e b saranno diversi.

Mi serve:
