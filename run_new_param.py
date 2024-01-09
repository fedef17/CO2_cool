#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

#import pickle

import new_param_lib_light as npl
#import yaml

###############################################################

# # open configuration file
# with open('config.yml', 'r', encoding='utf-8') as file:
#     cfg = yaml.load(file, Loader=yaml.FullLoader)

# outdir = cfg['dirs']['figs']
# ctl.mkdir(outdir)

# expdir = cfg['dirs']['exps']

###########################

vfit = 'vf5'
afit = 'a0s'
n_top = 65

thisdir = os.path.dirname(os.path.abspath(__file__)) + '/'

if not os.path.exists(thisdir + 'data/'):
    raise ValueError('data dir not found in data/')

ctag = '{}-{}-{}'.format(vfit, afit, n_top)
# coeff_file = thisdir + 'data/coeffs_finale_{}.p'.format(ctag)
# coeffs = pickle.load(open(coeff_file, 'rb'))
# print(coeffs.keys())
# for ke in coeffs.keys():
#     print(ke, coeffs[ke].shape)
#     np.save(thisdir + 'data/coeffs_{}_{}.npy'.format(ctag, ke), coeffs[ke])

coeffs = dict()
for ke in ['uco2', 'bsurf', 'co2profs', 'asurf', 'Lesc', 'alpha', 'acoeff', 'bcoeff']:
    coeffs[ke] = np.load(thisdir + 'data/coeffs_{}_{}.npy'.format(ctag, ke))

#   Alt     X        T          O         O2         N2        CO2         O3    AM
input_file = thisdir + 'input.dat'
datamat = np.loadtxt(input_file, comments = '#')

alts = datamat[:, 0]
X = datamat[:, 1]

pres = 1000.*np.exp(-X)

temp = datamat[:, 2]
ovmr = datamat[:, 3]
o2vmr = datamat[:, 4]
n2vmr = datamat[:, 5]
co2vmr = datamat[:, 6]

interp_coeffs = npl.precalc_interp_v1(coeffs = coeffs, n_top = n_top)

cr_new = npl.new_param_full_allgrids_v1(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, n_top = n_top)

outfile = thisdir + 'output.dat'

comment = '# Output of new param: Alt (km), X, pres (hPa), CR (K/day)\n# \n'
print(comment)

with open(outfile, 'w') as ofil:
    ofil.write(comment)
    for al, xi, pr, cr in zip(alts, X, pres, cr_new):
        strin = '{:8.2f} {:8.2f} {:12.3e} {:10.2f}\n'
        print(strin.format(al, xi, pr, cr))
        ofil.write(strin.format(al, xi, pr, cr))