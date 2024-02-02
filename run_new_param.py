#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import argparse

import new_param_lib_light as npl
import yaml

import warnings
###############################################################

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Command line argument example')

mod_rates = False
surf_temp = None
parser.add_argument('input_file')
parser.add_argument('--mod_rates', dest = 'mod_rates', action='store_true', default = False)
parser.add_argument('--surf_temp', dest = 'surf_temp', action='store', default = None)
parser.add_argument('--output', dest = 'output', action='store', default = 'output.dat')

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

# Load coefficients tables
ctag = '{}-{}-{}'.format(vfit, afit, n_top)
coeffs = dict()
for ke in ['uco2', 'bsurf', 'co2profs', 'asurf', 'Lesc', 'alpha', 'acoeff', 'bcoeff']:
    coeffs[ke] = np.load(thisdir + 'data/coeffs_{}_{}.npy'.format(ctag, ke))

# Load input atmosphere
datamat = np.loadtxt(input_file, comments = '#')
pres = datamat[:, 0]
X = np.log(1000./pres)
temp = datamat[:, 1]
co2vmr = datamat[:, 2]
ovmr = datamat[:, 3]
o2vmr = datamat[:, 4]
n2vmr = datamat[:, 5]

#with warnings.catch_warnings():
warnings.simplefilter("ignore") # suppress runtime warning due to zeros in the coeff matrix (for i > n_top)

# Calculate interpolating functions for coefficients
interp_coeffs = npl.precalc_interp_v1(coeffs = coeffs, n_top = n_top)

# Run param
if args.surf_temp is not None:
    surf_temp = float(args.surf_temp)
    print(f'Set surf_temp: {surf_temp}')
else:
    surf_temp = temp[0]

cr_new = npl.new_param_full_allgrids_v1(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, n_top = n_top, **rates)

## Write to output file
outfile = thisdir + args.output
comment = '# Output of new param: X, pres (hPa), CR (K/day)\n# \n'
print(comment)

with open(outfile, 'w') as ofil:
    ofil.write(comment)
    for xi, pr, cr in zip(X, pres, cr_new):
        strin = '{:8.2f} {:12.3e} {:10.2f}\n'
        print(strin.format(xi, pr, cr))
        ofil.write(strin.format(xi, pr, cr))