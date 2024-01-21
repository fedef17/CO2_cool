#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import argparse
#import matplotlib.pyplot as plt

#import pickle

import new_param_lib_light as npl
import yaml

###############################################################

# if len(sys.argv) < 2:
#     print("Usage: python run_new_param.py input_file")
#     sys.exit(1)  # Exit with an error code

# input_file = sys.argv[1]
#input_file = thisdir + 'input.dat'

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Command line argument example')

mod_rates = False
parser.add_argument('input_file')
parser.add_argument('--mod_rates', dest = 'mod_rates', action='store_true', default = False)

args = parser.parse_args()
print(args)
input_file = args.input_file

rates = dict()
if args.mod_rates:
    print('Reading modified rates from config file')

    # open configuration file
    with open('mod_rates.yml', 'r', encoding='utf-8') as rfile:
        cfg = yaml.load(rfile, Loader=yaml.FullLoader)
        
        for co in ['a_zo', 'b_zo', 'g_zo', 'a_zn2', 'b_zn2', 'g_zn2', 'a_zo2', 'b_zo2', 'g_zo2']:
            if co in cfg:
                rates[co] = float(cfg[co])

#####################################################################

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

cr_new = npl.new_param_full_allgrids_v1(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, n_top = n_top, **rates)

outfile = thisdir + 'output.dat'

comment = '# Output of new param: Alt (km), X, pres (hPa), CR (K/day)\n# \n'
print(comment)

with open(outfile, 'w') as ofil:
    ofil.write(comment)
    for al, xi, pr, cr in zip(alts, X, pres, cr_new):
        strin = '{:8.2f} {:8.2f} {:12.3e} {:10.2f}\n'
        print(strin.format(al, xi, pr, cr))
        ofil.write(strin.format(al, xi, pr, cr))