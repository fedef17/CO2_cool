#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt

#import climtools_lib as ctl

from scipy import io
import scipy.constants as const
import pickle

sys.path.insert(0, '/home/fedefab/Scrivania/Research/Dotto/Git/spect_robot/')
sys.path.insert(0, '/home/fedefab/Scrivania/Research/Dotto/Git/pythall/')
import spect_base_module as sbm
import spect_classes as spcl

kbc = const.k/(const.h*100*const.c) # 0.69503
#############################################################
###################################################################

cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'

# Calcolo le strengths per ogni banda (fundamental, first hot, second hot) in ogni intervallo spettrale
frini = 540.
frfin = 800.
all_freqs = np.arange(frini, frfin+1, 10.)

# leggo linee da hitran
# hitran_db = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/hitran2016_c02_450-900cm-1.par'
# lines_all = spcl.read_line_database(hitran_db, mol = 2, iso = None, up_lev = None, down_lev = None, fraction_to_keep = 0.1, db_format = 'HITRAN', freq_range = (frini, frfin), n_skip = 0, link_to_isomolecs = None, verbose = True)
#
# co2 = sbm.Molec(2, 'CO2')
# co2.add_all_iso_from_HITRAN(lines_all, n_max = 4, add_levels = True)
#
# for isonam in co2.all_iso:
#     print(isonam)
#     iso = getattr(co2, isonam)
#     for lev in iso.levels:
#         levvo = getattr(iso, lev)
#         print(lev, levvo.minimal_level_string(), levvo.energy)
#
# pickle.dump([co2, lines_all], open(cart_out + 'lines_and_mol.p', 'w'))
co2, lines_all = pickle.load(open(cart_out + 'lines_and_mol.p'))


spect_ranges = [(fr1, fr2) for fr1, fr2 in zip(all_freqs[:-1], all_freqs[1:])]

fun = ['0 0 0 0']
v1s = ['0 1 1 0']
v2s = ['0 2 2 0', '1 0 0 0']
v3s = ['0 3 3 0', '1 1 1 0']

T = 250.

all_total_strengths = dict()
all_hitran_strengths = dict()
all_bands = ['fun', '1st hot', '2nd hot'] #[(v1, fun), (v2, v1), (v3, v2)]
all_trans = dict()
all_trans['fun'] = (v1s, fun)
all_trans['1st hot'] = (v2s, v1s)
all_trans['2nd hot'] = (v3s, v2s)

all_temps = np.arange(140, 321, 5)

for band in all_bands:
    for fr1, fr2 in spect_ranges:
        lines_ok = [lin for lin in lines_all if (lin.Freq >= fr1) and (lin.Freq < fr2)]
        for T in all_temps:
            print(band, T, fr1)
            #print(len(lines_ok))

            for iso in range(1,5):
                #lines_band = [lin for lin in lines_ok if lin.minimal_level_string_up() == trans[0] and lin.minimal_level_string_lo() == trans[1] and lin.Iso == iso]
                lines_band = [lin for lin in lines_ok if lin.minimal_level_string_up() in all_trans[band][0] and lin.minimal_level_string_lo() in all_trans[band][1] and lin.Iso == iso]

                if T == all_temps[0]:
                    all_hitran_strengths[(band, (fr1, fr2), iso)] = np.sum([lin.Strength for lin in lines_band])
                #print(len(lines_band))

                strengths = [lin.CalcStrength(T) for lin in lines_band]
                all_total_strengths[(band, (fr1, fr2), T, iso)] = np.sum(strengths)
                #print('strng', all_total_strengths[(band, (fr1, fr2), T, iso)])
            if T == all_temps[0]:
                all_hitran_strengths[(band, (fr1, fr2), 'all')] = np.sum([all_hitran_strengths[(band, (fr1, fr2), iso)] for iso in range(1,5)])
            all_total_strengths[(band, (fr1, fr2), T, 'all')] = np.sum([all_total_strengths[(band, (fr1, fr2), T, iso)] for iso in range(1,5)])

for T in all_temps:
    for band in all_bands:
        for iso in range(1,5) + ['all']:
            if T == all_temps[0]:
                all_hitran_strengths[(band, 'full_band', iso)] = np.sum([all_hitran_strengths[(band, (fr1, fr2), iso)] for fr1,fr2 in spect_ranges])
            all_total_strengths[(band, 'full_band', T, iso)] = np.sum([all_total_strengths[(band, (fr1, fr2), T, iso)] for fr1,fr2 in spect_ranges])
            print(T, band, iso, all_total_strengths[(band, 'full_band', T, iso)])

pickle.dump([all_total_strengths, all_hitran_strengths], open(cart_out + 'line_strengths_CO2.p', 'w'))

all_total_strengths, all_hitran_strengths = pickle.load(open(cart_out + 'line_strengths_CO2.p'))

colors = ['indianred', 'forestgreen', 'steelblue']

plt.ion()
fig = plt.figure()
plt.yscale('log')
plt.ylim(1.e-23,1.e-17)
linestyle = ['-', '--', '-.', ':']
for iso, ls in zip(range(1,5), linestyle):
    for band, col in zip(all_bands, colors):
        strenga = [all_total_strengths[(band, 'full_band', T, iso)] for T in all_temps]
        plt.plot(all_temps, strenga, color = col, linestyle = ls, linewidth = 2)

plt.xlabel('Temp (K)')
plt.ylabel('Total band strength')
fig.savefig(cart_out + 'streng_full_band.pdf')

T = 300
linestyle = ['-', '--', ':']
for iso, ls in zip(range(1,5), linestyle):
    fig = plt.figure()
    plt.yscale('log')
    for band, col in zip(all_bands, colors):
        strenga = [all_total_strengths[(band, spran, T, iso)] for spran in spect_ranges]
        sprmid = [np.mean([fr1, fr2]) for fr1, fr2 in spect_ranges]
        plt.plot(sprmid, strenga, color = col, linestyle = ls, linewidth = 2, label = band)

    plt.ylim(1.e-23, None)
    plt.legend()
    plt.title('iso {} at {} K'.format(iso, T))
    fig.savefig(cart_out + 'streng_iso_{}.pdf'.format(iso))
