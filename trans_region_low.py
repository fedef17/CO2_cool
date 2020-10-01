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
    cart_out_2 = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE/'

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

ratiooo = pickle.load(open(cart_out_2 + 'ratios_NLTE_smooth.p', 'rb'))

# Now. We are in the low transition region and need to adjust the LTE coeffs to non-LTE.
cartsav = '/home/fabiano/Research/lavori/CO2_cooling/new_param/sav_v3.2/'
filsav = 'cr_nlte_{}_co2_{}.sav'
all_coeffs_nlte = dict()

for cco2 in range(1,8):
    for atm in allatms:
        coso = io.readsav(cartsav+filsav.format(atm, cco2))['data']
        nomi = 'HR_NLTE HR_NLTE_FB HR_NLTE_HOT HR_NLTE_ISO HR_LTE CO2_VMR O_VMR UCO2 L_ESC L_ESC_FOM'
        nomi = nomi.split()
        for nom in nomi:
            vara = getattr(coso, nom)[0]
            if 'HR' in nom:
                all_coeffs_nlte[(atm, cco2, nom.lower())] = -vara
            else:
                all_coeffs_nlte[(atm, cco2, nom.lower())] = vara

# per ogni atm faccio:
for cco2 in range(1,8):
    for atm in allatms:
        #print(atm, cco2)
        # for cnam in ['acoeff', 'bcoeff']:
        #     all_coeffs_nlte[(atm, cco2, cnam+'_new')] = all_coeffs[(atm, cco2, cnam)]*ratiooo[(atm, cco2, 'new_'+cnam[0])][np.newaxis, :]
        # for cnam in ['asurf', 'bsurf']:
        #     all_coeffs_nlte[(atm, cco2, cnam+'_new')] = all_coeffs[(atm, cco2, cnam)]*ratiooo[(atm, cco2, 'new_'+cnam[0])]
        #
        # for cnam in ['acoeff', 'bcoeff']:
        #     all_coeffs_nlte[(atm, cco2, cnam)] = all_coeffs[(atm, cco2, cnam)]*ratiooo[(atm, cco2, 'fomi')][np.newaxis, :]
        # for cnam in ['asurf', 'bsurf']:
        #     all_coeffs_nlte[(atm, cco2, cnam)] = all_coeffs[(atm, cco2, cnam)]*ratiooo[(atm, cco2, 'fomi')]

        hr_nlte = all_coeffs_nlte[(atm, cco2, 'hr_nlte')]
        hr_nlte_fun = all_coeffs_nlte[(atm, cco2, 'hr_nlte_fb')]+all_coeffs_nlte[(atm, cco2, 'hr_nlte_iso')]
        hr_nlte_hot = all_coeffs_nlte[(atm, cco2, 'hr_nlte_hot')]
        hr_lte = all_coeffs_nlte[(atm, cco2, 'hr_lte')]

        hr_lte_old = all_coeffs[(atm, cco2, 'hr_ref')]

        ksk = np.sum(np.abs(hr_lte_old-hr_lte))
        if ksk < 0.01:
            print(cco2, atm, 'OK!')
        else:
            print('---------> ', cco2, atm, 'NOUUUUUUU [{:.0f}]'.format(ksk), np.mean(hr_lte_old), np.mean(hr_lte))

            if cco2 == 4:
                hr_lte_old = all_coeffs[(atm, 5, 'hr_ref')]
                ksk = np.sum(np.abs(hr_lte_old-hr_lte))
                if ksk < 0.01:
                    print('-----------------> ok!! this is crazyyyy')
            elif cco2 == 5:
                hr_lte_old = all_coeffs[(atm, 4, 'hr_ref')]
                ksk = np.sum(np.abs(hr_lte_old-hr_lte))
                if ksk < 0.01:
                    print('-----------------> ok!! this is crazyyyy')

        hr_lte_fun, hr_lte_hot = npl.hr_LTE_FB_vs_ob(atm, cco2)

        for cnam in ['acoeff', 'bcoeff']:
            all_coeffs_nlte[(atm, cco2, cnam+'_new')] = all_coeffs[(atm, cco2, cnam)]*(hr_nlte_fun/hr_lte_fun)[np.newaxis, :]
        for cnam in ['asurf', 'bsurf']:
            all_coeffs_nlte[(atm, cco2, cnam+'_new')] = all_coeffs[(atm, cco2, cnam)]*(hr_nlte_hot/hr_lte_hot)

        for cnam in ['acoeff', 'bcoeff']:
            all_coeffs_nlte[(atm, cco2, cnam)] = all_coeffs[(atm, cco2, cnam)]*(hr_nlte/hr_lte)[np.newaxis, :]
        for cnam in ['asurf', 'bsurf']:
            all_coeffs_nlte[(atm, cco2, cnam)] = all_coeffs[(atm, cco2, cnam)]*(hr_nlte/hr_lte)

pickle.dump(all_coeffs_nlte, open(cart_out_2 + 'all_coeffs_NLTE.p', 'wb'))
all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

# Poi devo ripetere la roba di fomi_multiatmco2_ab_LTE_v4 per a e b separatamente.
# Cioè. tengo b della singola atm, b_nlte, e fitto i weights delle diverse atms per a.
# con questi weights ottengo un a per tutte le atm. Tengo ora fisso questo a e cerco i weights per b. Quindi i weights di a e b saranno diversi.
