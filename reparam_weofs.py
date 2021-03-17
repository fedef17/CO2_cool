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

if os.uname()[1] == 'ff-clevo':
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/SpectRobot/')
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/pythall/')
    cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'
else:
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'
    cart_out_2 = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE/'
    cart_out_rep = '/home/fabiano/Research/lavori/CO2_cooling/new_param/NLTE_reparam/'

import newparam_lib as npl
from eofs.standard import Eof

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

pres = atm_pt[('mle', 'pres')]
x = np.log(1000./pres)
n_alts_trlo = np.sum(x < 12.5)
print('low trans at {}'.format(alts[n_alts_trlo]))

all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

################################################################################

temps = [atm_pt[(atm, 'temp')] for atm in allatms]
temps = np.stack(temps)

temps_anom = np.stack([atm_pt[(atm, 'temp')]-np.mean(atm_pt[(atm, 'temp')]) for atm in allatms])
atm_anom_mean = np.mean(temps_anom, axis = 0)

solver = Eof(temps)
solver_anom = Eof(temps_anom)

fig = plt.figure()
for i, eo in enumerate(solver.eofs()):
    plt.plot(eo, all_alts, label = i)
plt.legend()
fig.savefig(cart_out_rep + 'eofs_temps.pdf')

fig = plt.figure()
for i, eo in enumerate(solver_anom.eofs()):
    plt.plot(eo, all_alts, label = i)
plt.legend()
fig.savefig(cart_out_rep + 'eofs_temps_anom.pdf')

fig = plt.figure()
plt.bar(np.arange(6), solver.eigenvalues())
fig.savefig(cart_out_rep + 'eigenvalues_temps.pdf')

fig = plt.figure()
plt.bar(np.arange(6), solver_anom.eigenvalues())
fig.savefig(cart_out_rep + 'eigenvalues_temps_anom.pdf')

fig = plt.figure()
plt.bar(np.arange(6), solver.varianceFraction())
fig.savefig(cart_out_rep + 'varfrac_temps.pdf')

fig = plt.figure()
plt.bar(np.arange(6), solver_anom.varianceFraction())
fig.savefig(cart_out_rep + 'varfrac_temps_anom.pdf')


fig = plt.figure()
atm_mean = np.mean(temps, axis = 0)
for i, pc in enumerate(solver.pcs()[:,0]):
    plt.plot(atm_mean+pc*solver.eofs()[0]-temps[i, :], all_alts)
fig.savefig(cart_out_rep + 'residual_temps_firstpc.pdf')

fig = plt.figure()
atm_mean = np.mean(temps, axis = 0)
for i, pc in enumerate(solver_anom.pcs()[:,0]):
    plt.plot(atm_anom_mean+pc*solver_anom.eofs()[0]-temps_anom[i, :], all_alts)
fig.savefig(cart_out_rep + 'residual_temps_anom_firstpc.pdf')

# plt.figure()
# for i, pc in enumerate(solver.pcs()[:,:2]):
#     plt.plot(atm_mean+pc[0]*solver.eofs()[0]+pc[1]*solver.eofs()[1]-temps[i,:], all_alts)


# ok so, if keeping only first and second eof I'm able to explain quite a fraction of the variability
# the coeffs will be written as: C = C0 + alpha*C1 + beta*C2, with C1 and C2 being the pcs of the actual temp profile with respect to the first two eofs. Calculation of C1 and C2 implies two dot products over 66 altitudes. Plus the sum to determine C. Affordable? yes!

# Now for the coeffs. Are the coeffs pcs linked to the temp pcs? (correlation?). If so, the method could work well!
cco2 = 7

surftemps = np.array([atm_pt[(atm, 'surf_temp')] for atm in allatms])

coefsolv = dict()
for conam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
    acos = np.stack([all_coeffs_nlte[(atm, cco2, conam)] for atm in allatms])
    aco_solver = Eof(acos)

    fig = plt.figure()
    plt.bar(np.arange(6), aco_solver.varianceFraction())
    fig.savefig(cart_out_rep + 'varfrac_{}.pdf'.format(conam))

    if 'coeff' in conam:
        cor0 = np.corrcoef(solver.pcs()[:, 0], aco_solver.pcs()[:, 0])[1,0]
        cor1 = np.corrcoef(solver.pcs()[:, 1], aco_solver.pcs()[:, 1])[1,0]
        cor0_anom = np.corrcoef(solver_anom.pcs()[:, 0], aco_solver.pcs()[:, 0])[1,0]
        cor1_anom = np.corrcoef(solver_anom.pcs()[:, 1], aco_solver.pcs()[:, 1])[1,0]
        print(conam, cor0, cor1, cor0_anom, cor1_anom)
    else:
        cor0 = np.corrcoef(surftemps, aco_solver.pcs()[:, 0])[1,0]
        cor1 = np.corrcoef(surftemps, aco_solver.pcs()[:, 1])[1,0]
        print(conam, cor0, cor1)
    coefsolv[conam] = aco_solver

# ('acoeff', 0.9201600549720309, 0.5650724813187429)
# ('bcoeff', 0.8852668113273987, 0.40514400023917907)
# ('asurf', -0.9916467503397747, 0.12197028746306282)
# ('bsurf', -0.9864472297843829, 0.14140499211950414)

# the scalar products between the temp anomalies and the first eof of the temperature profile
dotprods = np.array([np.dot(te-atm_anom_mean, solver_anom.eofs(eofscaling=1)[0]) for te in temps_anom])

colors = npl.color_set(6)

for conam in ['acoeff', 'bcoeff']:
    print(conam)
    print(np.corrcoef(dotprods, coefsolv[conam].pcs(pcscaling=1)[:, 0]))

    fig = plt.figure()
    for dotp, pcc, atm, col in zip(dotprods, coefsolv[conam].pcs(pcscaling=1)[:, 0], allatms, colors):
        plt.scatter([dotp], [pcc], color = col, label = atm)
    m, c, err_m, err_c = npl.linear_regre_witherr(dotprods, coefsolv['acoeff'].pcs(pcscaling=1)[:, 0])
    xcoso = np.linspace(-2, 2, 10)
    plt.plot(xcoso, m*xcoso + c)
    print(c, m)
    plt.xlabel('first pc of {}'.format(conam))
    plt.ylabel('projection of temp. anom on first eof')
    plt.legend()
    #npl.custom_legend(fig, colors, allatms, ncol = 3)
    fig.savefig(cart_out_rep + 'fit_{}_vs_tempeof.pdf'.format(conam))


surfanom = surftemps-np.mean(surftemps)
for conam in ['asurf', 'bsurf']:
    print(conam)
    print(np.corrcoef(surfanom, coefsolv[conam].pcs(pcscaling=1)[:, 0]))

    fig = plt.figure()
    for dotp, pcc, atm, col in zip(surfanom, coefsolv[conam].pcs(pcscaling=1)[:, 0], allatms, colors):
        plt.scatter([dotp], [pcc], color = col, label = atm)
    m, c, err_m, err_c = npl.linear_regre_witherr(surfanom, coefsolv[conam].pcs(pcscaling=1)[:, 0])
    xcoso = np.linspace(np.min(surfanom), np.max(surfanom), 10)
    plt.plot(xcoso, m*xcoso + c)
    print(c, m)
    plt.xlabel('first pc of {}'.format(conam))
    plt.ylabel('surf. temp. anomaly')
    plt.legend()
    #npl.custom_legend(fig, colors, allatms, ncol = 3)
    fig.savefig(cart_out_rep + 'fit_{}_vs_surftemp.pdf'.format(conam))
