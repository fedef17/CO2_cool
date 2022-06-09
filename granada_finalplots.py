#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pickle
from matplotlib import pyplot as plt

from scipy.interpolate import PchipInterpolator as spline
import numpy as np
import os
from scipy import io
from eofs.standard import Eof

import newparam_lib as npl

########################################################
if os.uname()[1] == 'xaru':
    sys.path.insert(0, '/home/fedef/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fedef/Research/git/pythall/')
    cart_base = '/home/fedef/Research/lavori/CO2_cooling/new_param/'
elif os.uname()[1] == 'hobbes':
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_base = '/home/fabiano/Research/lavori/CO2_cooling/new_param/'

import spect_base_module as sbm

cart_in = cart_base + '../MIPAS_2009/'

############################

cart_out = cart_base + 'LTE/'
cart_out_2 = cart_base + 'NLTE/'
cart_out_rep = cart_base + 'NLTE_reparam/'
cart_out_3 = cart_base + 'NLTE_upper/'
cart_out_F = cart_base + 'newpar_allatm_2/'

allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
#atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweights = np.ones(6)/6.
atmweights = dict(zip(allatms, atmweights))
allco2 = np.arange(1,npl.n_co2prof+1)

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v4.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt_v4.p'))

all_alts = atm_pt[('mle', 'alts')]

pres = atm_pt[('mle', 'pres')]
x_ref = np.log(1000./pres)

all_coeffs_nlte = pickle.load(open(cart_out_2 + 'all_coeffs_NLTE.p', 'rb'))

cose_upper_atm = pickle.load(open(cart_out_3 + 'cose_upper_atm.p', 'rb'))

alt2 = 51
n_top = 65
############################

cart_out_mip = cart_base + 'mipas_check/'

ctag = 'vf4-a1'
new_param_check = pickle.load(open(cart_out_mip+'check_all_out_ssw2009_{}.p'.format(ctag),'rb'))

inputs = pickle.load(open(cart_out_mip+'in_ssw2009_{}.p'.format('v10-nl0-65'),'rb'))

# inputs_rg = dict()
# for nam in ['temp', 'pres', 'ovmr', 'co2vmr', 'o2vmr', 'n2vmr', 'cr_mipas']:
#     inputs_rg[nam] = []
#
# for i in range(len(inputs['temp'])):
#     print(i)
#     x = inputs['x'][i]
#     for nam in ['temp', 'pres', 'ovmr', 'co2vmr', 'o2vmr', 'n2vmr', 'cr_mipas']:
#         if nam == 'pres':
#             spl = spline(x, np.log(inputs[nam][i]), extrapolate = False)
#             v2 = np.exp(spl(x_ref))
#         else:
#             spl = spline(x, inputs[nam][i], extrapolate = False)
#             v2 = spl(x_ref)
#         inputs_rg[nam].append(v2)
#
# for ke in inputs_rg:
#     inputs_rg[ke] = np.stack(inputs_rg[ke])
#
# pickle.dump(inputs_rg, open(cart_out_mip+'inrg_ssw2009-02-15.p','wb'))
inputs_rg = pickle.load(open(cart_out_mip+'inrg_ssw2009-02-15.p','rb'))

allntops = [57, 60, 63, 65, 67, 70]
allatms = npl.allatms
alt2 = 51
n_top = 65

#########################################################

###### PLOT sketch param
fig, ax = plt.subplots()
plt.plot(inputs_rg['temp'].T, all_alts, color = 'grey', lw = 0.3)
plt.plot(inputs_rg['temp'][0], all_alts, color = 'grey', lw = 0.3, label = 'MIPAS 15-02-2009')
for atm in allatms: plt.plot(atm_pt[(atm, 'temp')], all_alts, lw = 2, label = atm)
plt.legend()
plt.grid()
plt.xlabel('Temperature (K)')
ax.set_ylabel('Altitude')

ax.set_ylim(0., 200.)

ax2 = ax.twinx()
ax2.set_yticks(ax.get_yticks())

altsp = spline(x_ref, all_alts)
ax2ti = [altsp(xi) for xi in [0., 9.875, 12.625, 16.375, 20.125]]
ax2.set_yticks(ax2ti)
ax2.set_yticklabels([0., 9.875, 12.625, 16.375, 20.125])
ax2.set_ylabel('X')

for ti, col in zip(ax2ti[1:], ['indianred', 'gold', 'forestgreen', 'steelblue']):
    ax2.axhline(ti, color = col, ls = '--', lw = 2)

ax.legend(loc = 'lower right')

ti = ax.text(500, 55, 'LTE', fontsize = 18)

ti2 = ax.text(420, 75, 'NLTE low trans', fontsize = 16)
ti2.set_text('NLTE: modified Curtis matrix')
ti2.set_fontsize(14)
ti3 = ax.text(500, 92, 'NLTE upper trans', fontsize = 16)
ti3.set_text('NLTE: recurrence formula with alpha')
ti3.set_fontsize(14)

ti4 = ax.text(180, 160, 'NLTE: pure recurrence formula', fontsize = 14)
ti5 = ax.text(180, 185, 'NLTE: cool-to-space', fontsize = 14)

pio = np.array(ax.get_xlim())
iu = np.arange(pio[0], pio[1], 1)
ax.set_xlim(pio)
zood = ax.fill_between(iu, ax2ti[1], ax2ti[2], color = 'indianred', alpha = 0.1)
zood = ax.fill_between(iu, ax2ti[2], ax2ti[3], color = 'gold', alpha = 0.1)
zood = ax.fill_between(iu, ax2ti[3], ax2ti[4], color = 'forestgreen', alpha = 0.1)
zood = ax.fill_between(iu, ax2ti[4], 200., color = 'steelblue', alpha = 0.1)

fig.savefig(cart_out_mip + 'sketch_regions.pdf')

#####################################################
### plot weights LTE + lower transition region
## varfit weights
varfit_xis_5_nlte = pickle.load(open(cart_out_2+'varfit_NLTE_v5c.p', 'rb'))
varfit_xis_4_nlte = pickle.load(open(cart_out_2+'varfit_NLTE_v4c.p', 'rb'))

zuxi5 = np.stack([varfit_xis_5_nlte[(3, ii, 'afit')] for ii in range(55)])
wei5 = zuxi5/np.sum(zuxi5, axis = 1)[:, np.newaxis]
zuxi4 = np.stack([varfit_xis_4_nlte[(3, ii, 'afit')] for ii in range(55)])
wei4 = zuxi4/np.sum(zuxi4, axis = 1)[:, np.newaxis]

fig = plt.figure()
zuini = np.zeros(55)
for ii, zu in enumerate(wei5.T):
    plt.fill_betweenx(np.arange(55), zuini, zuini+zu, label = allatms[ii])
    print(ii, np.mean(zu))
    zuini = zuini + zu
plt.legend()
plt.title('vf5')
fig.savefig(cart_out_mip + 'check_finalweights_afit_vf5.pdf')

fig = plt.figure()
zuini = np.zeros(55)
for ii, zu in enumerate(wei4.T):
    plt.fill_betweenx(np.arange(55), zuini, zuini+zu, label = allatms[ii])
    print(ii, np.mean(zu))
    zuini = zuini + zu
plt.legend()
plt.title('vf4')
fig.savefig(cart_out_mip + 'check_finalweights_afit_vf4.pdf')


#####################################################

interp_coeffs_old = dict()
vfit = 'vf5'
for afit in ['a0s', 'a0']:
    for n_top in [57, 60, 63, 65, 67, 70]:
        print(n_top)
        ctag = '{}-{}-{}'.format(vfit, afit, n_top)
        coeff_file = cart_base + 'newpar_allatm/coeffs_finale_{}.p'.format(ctag)
        interp_coeffs_old[ctag] = npl.precalc_interp_old(coeff_file = coeff_file, n_top = n_top)

afit = 'a0s'
n_top = 75
for afi in ['a{}s'.format(i) for i in range(5)]:
    ctag = '{}-{}-{}'.format(vfit, afi, n_top)
    coeff_file = cart_base + 'newpar_allatm/coeffs_finale_{}.p'.format(ctag)
    interp_coeffs_old[ctag] = npl.precalc_interp_old(coeff_file = coeff_file, n_top = n_top)


def calc_all_refs(cco2 = 3, n_top = 65, debug_alpha = None, interp_coeffs = interp_coeffs_old, use_fomi = False, debug = False, extrap_co2col = False):
    """
    Calcs difference to all reference atms.
    """
    ref_calcs = []
    debucose = []

    for atm in allatms:
        #calc
        ii = allatms.index(atm)

        temp = atm_pt[(atm, 'temp')]
        surf_temp = atm_pt[(atm, 'surf_temp')]
        pres = atm_pt[(atm, 'pres')]

        hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_ref')]

        x_ref = np.log(1000./pres)

        co2vmr = atm_pt[(atm, cco2, 'co2')]
        ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
        o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
        n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]


        # L_esc = cose_upper_atm[(atm, cco2, 'L_esc_all_extP')]
        # lamb = cose_upper_atm[(atm, cco2, 'lamb')]
        # MM = cose_upper_atm[(atm, cco2, 'MM')]

        if use_fomi:
            alt_fomi, x_fomi, cr_fomi = npl.old_param(all_alts, temp, pres, co2vmr, Oprof = ovmr, O2prof = o2vmr, N2prof = n2vmr, input_in_ppm = False, cart_run_fomi = cart_run_fomi)
            spl = spline(x_fomi, cr_fomi)
            hr_calc = spl(x_ref)
        else:
            res = npl.new_param_full_allgrids(temp, temp[0], pres, co2vmr, ovmr, o2vmr, n2vmr, interp_coeffs = interp_coeffs, old_param = True, debug_alpha = debug_alpha, n_top = n_top, debug = debug, extrap_co2col = extrap_co2col)
            if debug:
                hr_calc, de = res
                debucose.append(de)
            else:
                hr_calc = res

        ref_calcs.append(hr_calc-all_coeffs_nlte[(atm, cco2, 'hr_ref')])

    ref_calcs = np.stack(ref_calcs)

    if not debug:
        return ref_calcs
    else:
        return ref_calcs, debucose

######

alpha_dic = pickle.load(open(cart_out_rep + 'alpha_unif_allwstart_ntops.p', 'rb'))

### check compare with brute-force alphas
# fig = plt.figure()
# for nto, col in zip(allntops, npl.color_set(len(allntops))):
#     #alphaok = np.append(alpha_dic[(afi, nto)]
#     afi = 'a0'
#     plt.axhline(x_ref[nto], ls = '--', lw = 0.5, color = col)
#     calcs = calc_all_refs(n_top = nto, interp_coeffs = interp_coeffs_old['vf5-a0-{}'.format(nto)])
#     rms = np.sqrt(np.mean(calcs**2, axis = 0))
#     plt.plot(rms[alt2-1:], x_ref[alt2-1:], label = nto, color = col)
#     calcs = calc_all_refs(n_top = nto, interp_coeffs = interp_coeffs_old['vf5-a0-{}'.format(nto)], debug_alpha = np.append(alpha_dic[(afi+'bf', nto)], 1))
#     rms = np.sqrt(np.mean(calcs**2, axis = 0))
#     plt.plot(rms[alt2-1:], x_ref[alt2-1:], color = col, ls = ':')
#
# plt.legend()
# plt.grid()

#### check compare with alpha of n_top 70 but truncated at n_top
# fig = plt.figure()
# for nto, col in zip(allntops, npl.color_set(len(allntops))):
#     #alphaok = np.append(alpha_dic[(afi, nto)]
#     afi = 'a0'
#     plt.axhline(x_ref[nto], ls = '--', lw = 0.5, color = col)
#     calcs = calc_all_refs(n_top = nto, interp_coeffs = interp_coeffs_old['vf5-a0-{}'.format(nto)])
#     rms = np.sqrt(np.mean(calcs**2, axis = 0))
#     plt.plot(rms[alt2-1:], x_ref[alt2-1:], label = nto, color = col)
#
#     #alpha_cos = np.append(alpha_dic[(afi, 70)][:nto+1-alt2], np.ones(70-nto))
#     alpha_cos = np.append(alpha_dic[(afi, 70)].squeeze()[:nto+1-alt2], np.ones(70-nto))
#     #calcs = calc_all_refs(n_top = nto, interp_coeffs = interp_coeffs_old['vf5-a0-{}'.format(nto)], debug_alpha = alpha_cos)
#     calcs = calc_all_refs(n_top = 70, interp_coeffs = interp_coeffs_old['vf5-a0-70'], debug_alpha = alpha_cos)
#     rms = np.sqrt(np.mean(calcs**2, axis = 0))
#     plt.plot(rms[alt2-1:], x_ref[alt2-1:], color = col, ls = ':')
#
# plt.legend()
# plt.grid()
#get_ipython().magic(u'save granada_day5_iaa.py 1-187')

#### check compare with new alphas start
fig = plt.figure()
for nto, col in zip(allntops, npl.color_set(len(allntops))):
    #alphaok = np.append(alpha_dic[(afi, nto)]
    afi = 'a0'
    plt.axhline(x_ref[nto], ls = '--', lw = 0.5, color = col)
    calcs = calc_all_refs(n_top = nto, interp_coeffs = interp_coeffs_old['vf5-{}-{}'.format(afi, nto)])
    rms = np.sqrt(np.mean(calcs**2, axis = 0))
    plt.plot(rms[alt2-1:], x_ref[alt2-1:], label = nto, color = col)

    afi = 'a0s'
    calcs = calc_all_refs(n_top = nto, interp_coeffs = interp_coeffs_old['vf5-{}-{}'.format(afi, nto)])
    rms = np.sqrt(np.mean(calcs**2, axis = 0))
    plt.plot(rms[alt2-1:], x_ref[alt2-1:], ls = ':', color = col)

afi = 'a0s'
calcs = calc_all_refs(n_top = 75, interp_coeffs = interp_coeffs_old['vf5-{}-{}'.format(afi, 75)])
rms = np.sqrt(np.mean(calcs**2, axis = 0))
plt.plot(rms[alt2-1:], x_ref[alt2-1:], ls = ':', color = 'grey', label = 75)

plt.legend()
plt.grid()

plt.ylabel('x')
plt.xlabel('RMS (K/day)')
plt.title('alpha fit: a0 (solid) vs a0s (:)')

fig.savefig(cart_out_mip + 'check_ref_RMS_alpha_a0vsa0s_allntop.pdf')

#### check compare all ntops a0s
fig = plt.figure()
for nto, col in zip(allntops+[75], npl.color_set(len(allntops)+1)):
    afi = 'a0s'
    plt.axhline(x_ref[nto], ls = '--', lw = 0.5, color = col)
    calcs = calc_all_refs(n_top = nto, interp_coeffs = interp_coeffs_old['vf5-{}-{}'.format(afi, nto)])
    rms = np.sqrt(np.mean(calcs**2, axis = 0))
    plt.plot(rms[alt2-1:], x_ref[alt2-1:], label = nto, color = col)

plt.legend()
plt.grid()
plt.title('alpha fit a0s: all n_tops')
plt.ylabel('x')
plt.xlabel('RMS (K/day)')

fig.savefig(cart_out_mip + 'check_ref_RMS_alpha_a0s_allntop.pdf')


#### check compare all ntops a0s
fig = plt.figure()
for nto, col in zip(allntops+[75], npl.color_set(len(allntops)+1)):
    afi = 'a0s'
    plt.axhline(x_ref[nto], ls = '--', lw = 0.5, color = col)
    calcs = calc_all_refs(n_top = nto, interp_coeffs = interp_coeffs_old['vf5-{}-{}'.format(afi, nto)])
    rms = np.sqrt(np.mean(calcs**2, axis = 0))
    plt.plot(rms[alt2-1:], x_ref[alt2-1:], label = nto, color = col)

    calcs = calc_all_refs(n_top = nto, interp_coeffs = interp_coeffs_old['vf5-{}-{}'.format(afi, nto)], extrap_co2col = False)
    rms = np.sqrt(np.mean(calcs**2, axis = 0))
    plt.plot(rms[alt2-1:], x_ref[alt2-1:], ls = ':', color = col)

plt.legend()
plt.grid()
plt.title('alpha fit a0s: all n_tops')
plt.ylabel('x')
plt.xlabel('RMS (K/day)')

fig.savefig(cart_out_mip + 'check_ref_RMS_alpha_a0s_allntop_noextrap.pdf')

#### check compare all ntops a0s
fig = plt.figure()
for nto, col in zip(allntops+[75], npl.color_set(len(allntops)+1)):
    afi = 'a0s'
    plt.axhline(x_ref[nto], ls = '--', lw = 0.5, color = col)
    calcs = calc_all_refs(n_top = nto, interp_coeffs = interp_coeffs_old['vf5-{}-{}'.format(afi, nto)])
    rms = np.sqrt(np.mean(calcs**2, axis = 0))
    plt.plot(rms[alt2-1:], x_ref[alt2-1:], label = nto, color = col)

calcs = calc_all_refs(use_fomi = True)
rms = np.sqrt(np.mean(calcs**2, axis = 0))
plt.plot(rms[alt2-1:], x_ref[alt2-1:], ls = '-.', color = 'black', label = 'fomi')

plt.legend()
plt.grid()
plt.title('alpha fit a0s: all n_tops')
plt.ylabel('x')
plt.xlabel('RMS (K/day)')

fig.savefig(cart_out_mip + 'check_ref_RMS_alpha_a0s_allntop_vsfomi.pdf')


#### check compare all ntops a0s alphas
fig = plt.figure()
for nto, col in zip((allntops+[75])[::-1], npl.color_set(len(allntops)+1)[::-1]):
    afi = 'a0s'
    plt.axhline(x_ref[nto], ls = '--', lw = 0.5, color = col)
    #plt.plot(np.append(alpha_dic[(afi, nto)][2], np.ones(25-(nto-alt2+1))), x_ref[alt2:76], color = col, label = nto)
    plt.plot(alpha_dic[(afi, nto)][2], x_ref[alt2:nto+1], color = col, label = nto)

plt.legend()
plt.grid()
plt.title('alpha fit a0s: all n_tops')

fig.savefig(cart_out_mip + 'check_ref_RMS_alpha_a0s_allntop_alphas.pdf')


#### check compare all ntops a0s
fig = plt.figure()
nto = 75
for afi, col in zip(['a{}s'.format(i) for i in range(5)], npl.color_set(5)):
    plt.axhline(x_ref[nto], ls = '--', lw = 0.5, color = col)
    calcs = calc_all_refs(n_top = nto, interp_coeffs = interp_coeffs_old['vf5-{}-{}'.format(afi, nto)])
    rms = np.sqrt(np.mean(calcs**2, axis = 0))
    plt.plot(rms[alt2-1:], x_ref[alt2-1:], label = afi, color = col)

calcs = calc_all_refs(use_fomi = True)
rms = np.sqrt(np.mean(calcs**2, axis = 0))
plt.plot(rms[alt2-1:], x_ref[alt2-1:], ls = '-.', color = 'black', label = 'fomi')

plt.legend()
plt.grid()
plt.title('alpha fit: all weightings')
plt.ylabel('x')
plt.xlabel('RMS (K/day)')

fig.savefig(cart_out_mip + 'check_ref_RMS_alpha_allafit_vsfomi.pdf')


#### check compare all ntops a0s alphas
fig = plt.figure()
for afi, col in zip(['a{}s'.format(i) for i in range(5)], npl.color_set(5)):
    plt.axhline(x_ref[nto], ls = '--', lw = 0.5, color = col)
    plt.plot(alpha_dic[(afi, nto)][2], x_ref[alt2:nto+1], color = col, label = afi)

plt.legend()
plt.grid()
plt.title('alpha fit: all weightings')

fig.savefig(cart_out_mip + 'check_ref_RMS_alpha_allafit_alphas.pdf')
