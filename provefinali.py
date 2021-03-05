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

coeffs = pickle.load(open(cart_out_4 + 'coeffs_finale.p', 'rb'))

####################################################################################

ialt_strano = 41
atm = 'mle'

interp_coeffs = dict()
for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf', 'alpha', 'Lesc']:
    int_fun, signc = npl.interp_coeff_logco2(coeffs[nam], coeffs['co2profs'])
    interp_coeffs[(nam, 'int_fun')] = int_fun
    interp_coeffs[(nam, 'signc')] = signc

print('Coeffs from interpolation!')
mults = np.arange(0.25, 8.1, 0.25)
#for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf', 'alpha', 'Lesc']:

colors = npl.color_set(n_alts_trlo)
for nam in ['acoeff', 'bcoeff']:
    figs = []
    axes = []

    namsurf = nam[0]+'surf'
    coefff = []
    for co2mult in mults:
        co2vmr = co2mult*atm_pt[(atm, 2, 'co2')]
        int_fun = interp_coeffs[(namsurf, 'int_fun')]
        sc = interp_coeffs[(namsurf, 'signc')]
        interplog = [int_fun[ialt](co2vmr[ialt]) for ialt in range(n_alts_trlo)]

        #print(co2mult, interplog)
        coefff.append(interplog)

    coefff = np.stack(coefff)

    fig, ax = plt.subplots()
    ax.grid()
    for j, col in zip(range(n_alts_trlo), colors):
        ax.plot(mults, coefff[:, j], color = col, linewidth = 0.5)
        #plt.scatter(coeffs['co2profs'][:, ialt]/coeffs['co2profs'][1, ialt], -np.log(coeffs[nam][:, j, ialt]/coeffs['co2profs'][:, ialt]))
    ax.set_title(namsurf)
    ax.set_xlabel('x CO2')

    figs.append(fig)
    axes.append(ax)

    for ialt in range(n_alts_trlo):
        coefff = []
        for co2mult in mults:
            co2vmr = co2mult*atm_pt[(atm, 2, 'co2')]
            int_fun = interp_coeffs[(nam, 'int_fun')]
            sc = interp_coeffs[(nam, 'signc')]
            if int_fun.ndim == 1:
                interplog = int_fun[ialt](co2vmr[ialt])
            else:
                interplog = np.array([intfu(co2vmr[ialt]) for intfu in int_fun[..., ialt]])
            print(co2mult, interplog)
            coefff.append(interplog)

        coefff = np.stack(coefff)
        colors = npl.color_set(n_alts_trlo)

        fig, ax = plt.subplots()
        ax.grid()
        for j, col in zip(range(n_alts_trlo), colors):
            ax.plot(mults, coefff[:, j], color = col, linewidth = 0.5)
            #plt.scatter(coeffs['co2profs'][:, ialt]/coeffs['co2profs'][1, ialt], -np.log(coeffs[nam][:, j, ialt]/coeffs['co2profs'][:, ialt]))
        ax.set_title(nam + '[:, {}]'.format(ialt))
        ax.set_xlabel('x CO2')

        figs.append(fig)
        axes.append(ax)

    npl.adjust_ax_scale(axes)
    npl.plot_pdfpages(cart_out_4 + 'check_{}_all.pdf'.format(nam), figs)
