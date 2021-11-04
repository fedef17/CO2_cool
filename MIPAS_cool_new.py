#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
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

if not os.path.exists(cart_out): os.mkdir(cart_out)

cart = '/home/fabiano/Research/lavori/CO2_cooling/MIPAS_2009/'

savT = io.readsav(cart+'CR20090215/L2_20090215_T_521.6', verbose=True)
savCR = io.readsav(cart+'CR20090215/L2_20090215_CR-CO2-IR_521.6', verbose=True)
savO = io.readsav(cart+'CR20090215/L2_20090215_O_521.6', verbose=True)
savCO2 = io.readsav(cart+'CR20090215/L2_20090215_CO2_521.6', verbose=True)

T = savT.result
CR = savCR.result
O = savO.result
CO2 = savCO2.result

# questo stampa i nomi degli ingressi (help di IDL)
print(CR.dtype.names)

alts = CR.altitude[1]

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
        ('temperature', 'O'), ('cr_mipas', 'O'), ('alt_fomi', 'O'), ('cr_fomi', 'O'), ('cr_fomi_int', 'O')]
res = np.empty(1, dtype = tipi)
res = res.view(np.recarray)

restot = res

# Prova 1: atmosfera polare media durante un SSW
alt_manuel, mol_vmrs, molist, molnums = sbm.read_input_vmr_man(cart + 'gases_120.dat', version = 2)
alt_manuel = np.linspace(0,120,121)

sys.exit()

# for il in range(2):
old_param = []
new_param = []
obs = []
for il in range(len(CR)):
    print(il)
    temp = T.target[il]
    pres = T.pressure[il]

    Ocon = O.target[il]
    splO = spline(alts, np.log(Ocon))
    Ocon = splO(alt_manuel)
    Ocon = np.exp(Ocon)

    CO2con = CO2.target[il]
    splCO2 = spline(alts, CO2con)
    CO2con = splCO2(alt_manuel)

    splT = spline(alts,temp)
    temp = splT(alt_manuel)

    splP = spline(alts,np.log(pres))
    pres = splP(alt_manuel)
    pres = np.exp(pres)

    filename = cart+'atm_manuel.dat'
    sbm.scriviinputmanuel(alt_manuel,temp,pres,filename)

    mol_vmrs['CO2'] = CO2con
    mol_vmrs['O'] = Ocon
    filename = cart+'vmr_atm_manuel.dat'
    sbm.write_input_vmr_man(filename, alt_manuel, mol_vmrs, hit_gas_list = molist, hit_gas_num = molnums, version = 2)

    call(cart+'./fomi_mipas')
    nomeout = cart+'output__mipas.dat'
    alt_fomi, cr_fomi = sbm.leggioutfomi(nomeout)

    cr_fomi = npl.old_param(alt_manuel, temp, pres, CO2con, Oprof = Ocon)
    cr_new = npl.new_param_full_allgrids(alt_manuel, temp, temp[0], pres, CO2con, Ocon, o2vmr, n2vmr)

    obs.append(res.cr_mipas[0])
    old_param.append(cr_fomi)
    new_param.append(cr_new)

    res.date[0] = CR.date[il]
    res.latitude[0] = CR.latitude[il]
    res.longitude[0] = CR.longitude[il]
    res.sza[0] = CR.sza[il]
    res.altitude[0] = CR.altitude[il]
    res.pressure[0] = CR.pressure[il]
    res.temperature[0] = CR.temperature[il]

    res.cr_mipas[0] = CR.target[il]
    res.alt_fomi[0] = alt_fomi
    res.cr_fomi[0] = cr_fomi

    splcr = spline(alt_fomi,cr_fomi)
    res.cr_fomi_int[0] = splcr(res.altitude[0])

    restot = np.append(restot,res)


restot = restot[1:]
restot = restot.view(np.recarray)

pickle.dump(restot, open(cart_out+'ssw2009_v3_okTOCO2_1e13_newparam.p','wb'))

# produco atmosfera di input in formato manuel ->
# lancio fomi
# leggo output e salvo in una struttura
