#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
import math as mt
import spect_base_module as sbm
from subprocess import call
import pickle
import scipy.io as io

import json

############################################

cart = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/MIPAS_2009/'

tag = '1e13'
restot = pickle.load(open(cart+'ssw2009_v3_okTOCO2_1e13.pyc'))
print(len(restot))
print(restot[0].dtype.names)
allnams = restot[0].dtype.names

n_res = range(len(restot))

allresdicts = []
for num in n_res:
    resdict = dict()
    for nam in allnams:
        coso = getattr(restot, nam)[num]
        if type(coso) is np.ndarray:
            coso = list(coso)
        elif type(coso) in [float, np.float32, np.float64]:
            coso = float(coso)
        resdict[nam] = coso
    allresdicts.append(resdict)

mat_dict = {nam : getattr(restot, nam) for nam in allnams}

# gigi = dict()
# gigi['ciao'] = 2
# gigi['due'] = 'ciao'
# gigi['lungo'] = [0.1, 0.24153531, 1.e-23]
# listago = [gigi, gigi]
#
# with open(cart + 'test.json', 'w') as outfile:
#     json.dump(listago, outfile)

listago = allresdicts
with open(cart + 'Mipas2009_fomi_{}.json'.format(tag), 'w') as outfile:
    json.dump(listago, outfile)

io.savemat(cart + 'Mipas2009_fomi_{}.mat'.format(tag), mdict = mat_dict)

pickle.dump(allresdicts, open(cart + 'Mipas2009_fomi_dict_{}.p'.format(tag), 'w'))

gigi = np.stack(restot.cr_fomi_int)
np.savetxt(cart + 'Mipas2009_fomiCR_{}.txt'.format(tag), gigi)

gigi = np.stack(restot.cr_mipas)
np.savetxt(cart + 'Mipas2009_mipasCR_{}.txt'.format(tag), gigi)

filo = open(cart + 'Mipas2009_fomi_auxinfo_{}.txt'.format(tag), 'w')
for i, cos in enumerate(allresdicts):
    dat = cos['date']
    lat = cos['latitude']
    lon = cos['longitude']
    sza = cos['sza']
    filo.write('{:3d} {:20s} {:8.2f} {:8.2f} {:8.2f}\n'.format(i+1, dat, lat, lon, sza))
filo.close()

#
# CR = io.readsav(cart+'L2_20090215_CR-CO2-IR_521', verbose=True).result
#
# alldats = CR.DATE
#
# date_not = []
# for dat in alldats:
#     if dat not in restot.date:
#         date_not.append(dat)
#         print(dat)
#
# print(len(date_not))
