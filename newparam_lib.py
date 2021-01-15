#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
import matplotlib.cm as cm
#import climtools_lib as ctl

from scipy import io
import scipy.constants as const
from scipy import interpolate
from scipy.interpolate import PchipInterpolator as spline

from subprocess import call

import pickle
import pandas as pd

if os.uname()[1] == 'ff-clevo':
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/SpectRobot/')
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/pythall/')
    cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'
elif os.uname()[1] == 'hobbes':
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'
else:
    raise ValueError('Unknown platform {}. Specify paths!'.format(os.uname()[1]))

import spect_base_module as sbm
import spect_classes as spcl

kbc = const.k/(const.h*100*const.c) # 0.69503
kboltz = 1.38064853e-23 # J/K
E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition

cp = 1.005e7 # specific enthalpy dry air - erg g-1 K-1

NLTE_DEBUG = True
#############################################################


allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweigths = dict(zip(allatms, atmweigths))
allco2 = np.arange(1,8)

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v2.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt_v2.p'))
n_alts = 40

from scipy.optimize import Bounds, minimize, least_squares

#############################################################


def new_param_LTE(interp_coeffs, temp, co2pr, surf_temp = None, tip = 'varfit'):
    """
    Calculates the new param, starting from interp_coeffs.
    """
    if surf_temp is None:
        surf_temp = temp[0]

    coeffs = []
    for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        int_fun = interp_coeffs[(tip, nam, 'int_fun')]
        sc = interp_coeffs[(tip, nam, 'signc')]

        coeffs.append(coeff_from_interp(int_fun, sc, co2pr))

    acoeff, bcoeff, asurf, bsurf = coeffs

    hr_calc = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)

    return hr_calc


def old_param(alts, temp, pres, CO2prof, cart_run_fomi = '/home/fabiano/Research/lavori/CO2_cooling/cart_run_fomi/'):
    """
    Run the old param.

    WARNING!!! CO2prof should be in concentration (not ppm!). Gets converted to ppm inside the routine.
    """

    fil_VMR = cart_run_fomi + 'gases_120.dat'
    alt_manuel, mol_vmrs, molist, molnums = sbm.read_input_vmr_man(fil_VMR, version = 2)

    splCO2 = spline(alts, CO2prof)
    CO2con = splCO2(alt_manuel)
    print('piooo', np.median(CO2con))

    splT = spline(alts, temp)
    temp = splT(alt_manuel)

    splP = spline(alts,np.log(pres))
    pres = splP(alt_manuel)
    pres = np.exp(pres)

    filename = cart_run_fomi + 'atm_manuel.dat'
    sbm.scriviinputmanuel(alt_manuel, temp, pres, filename)

    mol_vmrs['CO2'] = CO2con*1.e6
    filename = cart_run_fomi + 'vmr_atm_manuel.dat'
    sbm.write_input_vmr_man(filename, alt_manuel, mol_vmrs, hit_gas_list = molist, hit_gas_num = molnums, version = 2)

    wd = os.getcwd()
    os.chdir(cart_run_fomi)
    call('./fomi_mipas')
    os.chdir(wd)
    nomeout = cart_run_fomi + 'output__mipas.dat'
    alt_fomi, cr_fomi = sbm.leggioutfomi(nomeout)

    return alt_fomi, cr_fomi


def get_interp_coeffs(tot_coeff_co2):
    interp_coeffs = dict()
    for tip in ['unifit', 'varfit']:
        co2profs = [atm_pt[('mle', cco2, 'co2')] for cco2 in range(1,8)]

        for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
            coeffs = [tot_coeff_co2[(tip, nam, cco2)] for cco2 in range(1,8)]

            int_fun, signc = interp_coeff_logco2(coeffs, co2profs)
            interp_coeffs[(tip, nam, 'int_fun')] = int_fun
            interp_coeffs[(tip, nam, 'signc')] = signc

    return interp_coeffs


def hr_atm_calc(atm, cco2):
    """
    This is the reference LTE cooling rate for each atm/cco2.
    """
    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    acoeff = all_coeffs[(atm, cco2, 'acoeff')]
    bcoeff = all_coeffs[(atm, cco2, 'bcoeff')]
    asurf = all_coeffs[(atm, cco2, 'asurf')]
    bsurf = all_coeffs[(atm, cco2, 'bsurf')]

    hr = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)

    return hr

def hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = 51):
    """
    This is the LTE cooling rate given a certain set of a and b coefficients.
    """
    n_alts = len(temp)
    epsilon_ab_tot = np.zeros(n_alts, dtype = float)

    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    for xi in range(n_alts):
        epsilon_ab_tot[xi] = np.sum((acoeff[:max_alts, xi] + bcoeff[:max_alts, xi]* phi_fun[xi]) * phi_fun[:max_alts]) # il contributo della colonna
        epsilon_ab_tot[xi] += (asurf[xi] + bsurf[xi]* phi_fun[xi]) * phi_fun_g

    return epsilon_ab_tot


def running_mean(var, wnd, remove_nans = False, keep_length = False):
    """
    Performs a running mean (if multidim, the mean is done on the first axis).

    < wnd > : is the window length.
    """
    if var.ndim == 1:
        tempser = pd.Series(var)
        rollpi_temp = tempser.rolling(wnd, center = True).mean()
        if remove_nans:
            rollpi_temp = np.array(rollpi_temp)
            okvals = ~np.isnan(rollpi_temp)
            coso = rollpi_temp[okvals]
            if keep_length:
                for ii in range(wnd):
                    if np.isnan(rollpi_temp[ii]):
                        rollpi_temp[ii] = coso[0]
                    if np.isnan(rollpi_temp[-ii]):
                        rollpi_temp[-ii] = coso[-1]
            else:
                rollpi_temp = coso
    else:
        rollpi_temp = []
        for i in range(len(var)):
            if i-wnd//2 < 0 or i + wnd//2 > len(var)-1:
                if remove_nans: continue
                rollpi_temp.append(np.nan*np.ones(var[0].shape))
            else:
                rollpi_temp.append(np.mean(var[i-wnd//2:i+wnd//2+1, ...], axis = 0))

        rollpi_temp = np.stack(rollpi_temp)

    return rollpi_temp


def custom_legend(fig, colors, labels, loc = 'lower center', ncol = None, fontsize = 15, bottom_margin_per_line = 0.05):
    if ncol is None:
        ncol = int(np.ceil(len(labels)/2.0))
    plt.subplots_adjust(bottom = bottom_margin_per_line*np.ceil(1.0*len(labels)/ncol))
    proxy = [plt.Rectangle((0,0),1,1, fc = col) for col in colors]
    fig.legend(proxy, labels, loc = loc, ncol = ncol, fontsize = fontsize)
    return fig


def hr_LTE_FB_vs_ob(atm, cco2, max_alts = 51):
    """
    Gives the FB and rest-of-bands LTE CRs for a specific atm/cco2.
    """

    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    acoeff = all_coeffs[(atm, cco2, 'acoeff')]
    bcoeff = all_coeffs[(atm, cco2, 'bcoeff')]
    asurf = all_coeffs[(atm, cco2, 'asurf')]
    bsurf = all_coeffs[(atm, cco2, 'bsurf')]

    n_alts = len(temp)
    epsilon_FB = np.zeros(n_alts, dtype = float)
    epsilon_ob = np.zeros(n_alts, dtype = float)

    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    for xi in range(n_alts):
        epsilon_FB[xi] = np.sum(acoeff[:max_alts, xi] * phi_fun[:max_alts]) # il contributo della colonna
        epsilon_ob[xi] = np.sum(bcoeff[:max_alts, xi] * phi_fun[xi] * phi_fun[:max_alts]) # il contributo della colonna
        epsilon_FB[xi] += asurf[xi] * phi_fun_g
        epsilon_ob[xi] += bsurf[xi] * phi_fun[xi] * phi_fun_g

    return epsilon_FB, epsilon_ob


def hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, x0, max_alts = 51):
    """
    As above, but at a single altitude.
    """
    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    epsilon_ab_tot = np.sum((acoeff[:max_alts, x0] + bcoeff[:max_alts, x0]* phi_fun[x0]) * phi_fun[:max_alts]) # il contributo della colonna
    epsilon_ab_tot += (asurf[x0] + bsurf[x0]* phi_fun[x0]) * phi_fun_g

    return epsilon_ab_tot


def hr_from_xi(xis, atm, cco2, all_coeffs = all_coeffs, atm_pt = atm_pt, allatms = allatms):#, n_alts = 40):
    """
    Calculates the HR from the acoeff and bcoeff of the different atmospheres, using the weights xis.
    """

    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    hr_somma = np.zeros(len(temp), dtype = float)
    for atmprim, xi in zip(allatms, xis):
        acoeff = all_coeffs[(atmprim, cco2, 'acoeff')]
        bcoeff = all_coeffs[(atmprim, cco2, 'bcoeff')]
        asurf = all_coeffs[(atmprim, cco2, 'asurf')]
        bsurf = all_coeffs[(atmprim, cco2, 'bsurf')]

        h_ab = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)
        #print(atm, xi, np.mean(h_ab))
        hr_somma += xi * h_ab#[:n_alts]

    hr_somma = hr_somma/np.sum(xis)

    return hr_somma


def coeff_from_xi_at_x0(xis, cco2, ialt, cnam = None, all_coeffs = None, atm_pt = atm_pt, allatms = allatms):
    """
    Calculates the acoeff/asurf/bcoeff/bsurf from the respective coeffs of the different atmospheres, using the weights xis.
    """
    #print(xis, cco2, ialt)
    #xis = np.stack(xis)
    #print(xis.shape)

    if all_coeffs is None:
        raise ValueError('Specify all_coeffs to use (LTE or NLTE)')

    if cnam is None:
        raise ValueError('Specify coeff setting cnam')

    a_somma = 0. * all_coeffs[(allatms[0], 1, cnam)][..., 0]
    for atmprim, xi in zip(allatms, xis):
        acoeff = all_coeffs[(atmprim, cco2, cnam)][..., ialt]
        a_somma += xi * acoeff

    a_somma = a_somma/np.sum(xis)

    return a_somma


def hr_from_xi_at_x0(xis, atm, cco2, ialt, all_coeffs = all_coeffs, atm_pt = atm_pt, allatms = allatms):
    """
    Calculates the HR from the acoeff and bcoeff of the different atmospheres, using the weights xis.
    """

    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    hr_somma = 0.
    for atmprim, xi in zip(allatms, xis):
        acoeff = all_coeffs[(atmprim, cco2, 'acoeff')]
        bcoeff = all_coeffs[(atmprim, cco2, 'bcoeff')]
        asurf = all_coeffs[(atmprim, cco2, 'asurf')]
        bsurf = all_coeffs[(atmprim, cco2, 'bsurf')]

        h_ab = hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, ialt)
        #print(atm, xi, np.mean(h_ab))
        hr_somma += xi * h_ab

    hr_somma = hr_somma/np.sum(xis)

    return hr_somma


def hr_from_xi_at_x0_afit(xis, atm, cco2, ialt, xis_b, all_coeffs = all_coeffs, atm_pt = atm_pt, allatms = allatms, max_alts = 51):
    """
    Calculates the HR from the acoeff and bcoeff of the different atmospheres, using the weights xis. But applies the weights only to acoeffs, keeping b fixed.
    """
    if NLTE_DEBUG and ('tro', 1, 'hr_nlte') not in all_coeffs.keys():
        print(all_coeffs.keys())
        raise ValueError('NOT the right coeffs!')

    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    agn = coeff_from_xi_at_x0(xis, cco2, ialt, cnam = 'acoeff', all_coeffs = all_coeffs)
    agn_surf = coeff_from_xi_at_x0(xis, cco2, ialt, cnam = 'asurf', all_coeffs = all_coeffs)

    if xis_b is None:
        bcoeff = all_coeffs[(atm, cco2, 'bcoeff')][:, ialt]
        bsurf = all_coeffs[(atm, cco2, 'bsurf')][ialt]
    else:
        bcoeff = coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bcoeff', all_coeffs = all_coeffs)
        bsurf = coeff_from_xi_at_x0(xis_b, cco2, ialt, cnam = 'bsurf', all_coeffs = all_coeffs)

    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    epsilon_ab_tot = np.sum((agn[:max_alts] + bcoeff[:max_alts]* phi_fun[ialt]) * phi_fun[:max_alts]) # il contributo della colonna
    epsilon_ab_tot += (agn_surf + bsurf* phi_fun[ialt]) * phi_fun_g

    return epsilon_ab_tot


def hr_from_xi_at_x0_bfit(xis, atm, cco2, ialt, xis_a, all_coeffs = all_coeffs, atm_pt = atm_pt, allatms = allatms, max_alts = 51):
    """
    Calculates the HR from the acoeff and bcoeff of the different atmospheres, using the weights xis. But applies the weights only to acoeffs, keeping b fixed.
    """
    if NLTE_DEBUG and ('tro', 1, 'hr_nlte') not in all_coeffs.keys():
        print(all_coeffs.keys())
        raise ValueError('NOT the right coeffs!')

    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    bgn = coeff_from_xi_at_x0(xis, cco2, ialt, cnam = 'bcoeff', all_coeffs = all_coeffs)
    bgn_surf = coeff_from_xi_at_x0(xis, cco2, ialt, cnam = 'bsurf', all_coeffs = all_coeffs)

    if xis_a is None:
        acoeff = all_coeffs[(atm, cco2, 'acoeff')]
        asurf = all_coeffs[(atm, cco2, 'asurf')]
    else:
        acoeff = coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'acoeff', all_coeffs = all_coeffs)
        asurf = coeff_from_xi_at_x0(xis_a, cco2, ialt, cnam = 'asurf', all_coeffs = all_coeffs)

    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    epsilon_ab_tot = np.sum((acoeff[:max_alts] + bgn[:max_alts]* phi_fun[ialt]) * phi_fun[:max_alts]) # il contributo della colonna
    epsilon_ab_tot += (asurf + bgn_surf* phi_fun[ialt]) * phi_fun_g

    return epsilon_ab_tot


def ab_from_xi_unifit(xis, cco2, all_coeffs = all_coeffs, allatms = allatms):
    """
    Calculates the fitted acoeff and bcoeff, using the weights xis.
    """

    acoeff = all_coeffs[('mle', 1, 'acoeff')]
    acoefftot = np.zeros(acoeff.shape)
    bcoefftot = np.zeros(acoeff.shape)
    xiiis = xis/np.sum(xis)

    for atmprim, xi in zip(allatms, xiiis):
        acoeff = all_coeffs[(atmprim, cco2, 'acoeff')]
        bcoeff = all_coeffs[(atmprim, cco2, 'bcoeff')]
        # asurf = all_coeffs[(atmprim, cco2, 'asurf')]
        # bsurf = all_coeffs[(atmprim, cco2, 'bsurf')]

        acoefftot += xi*acoeff
        bcoefftot += xi*bcoeff

    return acoefftot, bcoefftot


def absurf_from_xi_unifit(xis, cco2, all_coeffs = all_coeffs, allatms = allatms):
    """
    Calculates the fitted acoeff and bcoeff, using the weights xis.
    """

    asurf = all_coeffs[('mle', 1, 'asurf')]
    asurftot = np.zeros(asurf.shape)
    bsurftot = np.zeros(asurf.shape)

    xiiis = xis/np.sum(xis)

    for atmprim, xi in zip(allatms, xiiis):
        asurf = all_coeffs[(atmprim, cco2, 'asurf')]
        bsurf = all_coeffs[(atmprim, cco2, 'bsurf')]

        asurftot += xi*asurf
        bsurftot += xi*bsurf

    return asurftot, bsurftot


def ab_from_xi_abfit_fromdict(xis_ab, cco2, all_coeffs = all_coeffs, allatms = allatms, faircoeff = False):
    """
    Calculates the fitted acoeff and bcoeff, using the weights xis_a and xis_b.

    xis is a dict with keys (cco2, ialt).

    NOTE!!! -------> faircoeff sets all xis to 1/6.
    """

    acoeff = all_coeffs[('mle', 1, 'acoeff')]
    nalt = acoeff.shape[1]

    xis_a_alts = [xis_ab[(cco2, ialt, 'afit')] for ialt in range(nalt)]
    xis_b_alts = [xis_ab[(cco2, ialt, 'bfit')] for ialt in range(nalt)]
    #print(xis_a_alts, xis_b_alts)
    if faircoeff:
        xis_a_alts = [1./6.*np.ones(6)]*nalt
        xis_b_alts = [1./6.*np.ones(6)]*nalt

    agn, bgn, agn_surf, bgn_surf = ab_from_xi_abfit(xis_a_alts, xis_b_alts, cco2, all_coeffs = all_coeffs, allatms = allatms)

    return agn, bgn, agn_surf, bgn_surf


def ab_from_xi_abfit(xis_a, xis_b, cco2, all_coeffs = all_coeffs, allatms = allatms, usevar = ''):
    """
    Calculates the fitted acoeff and bcoeff, using the weights xis_a and xis_b.

    xis is a dict with keys (cco2, ialt).
    """

    acoeff = all_coeffs[('mle', 1, 'acoeff')]
    agn = np.zeros(acoeff.shape)
    bgn = np.zeros(acoeff.shape)
    asurf = all_coeffs[('mle', 1, 'asurf')]
    agn_surf = np.zeros(asurf.shape)
    bgn_surf = np.zeros(asurf.shape)

    nalt = acoeff.shape[1]
    for xia, xib, ialt in zip(xis_a, xis_b, range(nalt)):
        acoeff = coeff_from_xi_at_x0(xia, cco2, ialt, cnam = 'acoeff'+usevar, all_coeffs = all_coeffs)
        asurf = coeff_from_xi_at_x0(xia, cco2, ialt, cnam = 'asurf'+usevar, all_coeffs = all_coeffs)
        bcoeff = coeff_from_xi_at_x0(xib, cco2, ialt, cnam = 'bcoeff'+usevar, all_coeffs = all_coeffs)
        bsurf = coeff_from_xi_at_x0(xib, cco2, ialt, cnam = 'bsurf'+usevar, all_coeffs = all_coeffs)

        agn[:, ialt] = acoeff
        agn_surf[ialt] = asurf
        bgn[:, ialt] = bcoeff
        bgn_surf[ialt] = bsurf

    return agn, bgn, agn_surf, bgn_surf


def ab_from_xi_varfit(xis, cco2, all_coeffs = all_coeffs, allatms = allatms):
    """
    Calculates the fitted acoeff and bcoeff, using the weights xis.

    xis is a dict with keys (cco2, ialt).
    """

    acoeff = all_coeffs[('mle', 1, 'acoeff')]
    acoefftot = np.zeros(acoeff.shape)
    bcoefftot = np.zeros(acoeff.shape)

    for ialt in range(acoeff.shape[1]):
        xiiis = xis[(cco2, ialt)]/np.sum(xis[(cco2, ialt)])
        for atmprim, xi in zip(allatms, xiiis):
            acoeff = all_coeffs[(atmprim, cco2, 'acoeff')]
            bcoeff = all_coeffs[(atmprim, cco2, 'bcoeff')]

            acoefftot[:, ialt] += xi*acoeff[:, ialt]
            bcoefftot[:, ialt] += xi*bcoeff[:, ialt]

    return acoefftot, bcoefftot


def absurf_from_xi_varfit(xis, cco2, all_coeffs = all_coeffs, allatms = allatms):
    """
    Calculates the fitted acoeff and bcoeff, using the weights xis.

    xis is a dict with keys (cco2, ialt).
    """

    asurf = all_coeffs[('mle', 1, 'asurf')]
    asurftot = np.zeros(asurf.shape)
    bsurftot = np.zeros(asurf.shape)

    for ialt in range(asurf.shape[0]):
        xiiis = xis[(cco2, ialt)]/np.sum(xis[(cco2, ialt)])
        for atmprim, xi in zip(allatms, xiiis):
            asurf = all_coeffs[(atmprim, cco2, 'asurf')]
            bsurf = all_coeffs[(atmprim, cco2, 'bsurf')]

            asurftot[ialt] += xi*asurf[ialt]
            bsurftot[ialt] += xi*bsurf[ialt]

    return asurftot, bsurftot


def linear_regre_witherr(x, y):
    """
    Makes a linear regression of dataset y in function of x using numpy.polyfit. Returns the coefficient m and c: y = mx + c. And their estimated error.
    """

    xord = np.argsort(x)
    x = x[xord]
    y = y[xord]

    res = np.polyfit(x, y, deg = 1, cov = True)
    m,c = res[0]
    covmat = res[1]

    err_m = np.sqrt(covmat[0,0])
    err_c = np.sqrt(covmat[1,1])

    return m, c, err_m, err_c


def linfit_coeff_logco2(coeffs, co2_profs):
    """
    Interpolates log(a/cco2)
    """

    ndim = coeffs[0].ndim
    m_coeff = np.zeros(coeffs[0].shape)
    errm_coeff = np.zeros(coeffs[0].shape)
    c_coeff = np.zeros(coeffs[0].shape)
    errc_coeff = np.zeros(coeffs[0].shape)

    sign_coeff = np.zeros(coeffs[0].shape)

    n_alts = coeffs[0].shape[0]

    for ialt in range(n_alts):
        co2p = np.array([co[ialt] for co in co2_profs])
        if ndim == 2:
            for j in range(n_alts):
                cval = np.array([co[j, ialt] for co in coeffs])

                if np.all(cval < 0):
                    print('All values are negative! at ({},{})\n'.format(j, ialt))
                    logcval = np.log(-cval/co2p)
                    sign_coeff[j, ialt] = -1
                elif np.any(cval < 0):
                    raise ValueError('Only some value is negative! at ({},{})'.format(j, ialt))
                else:
                    logcval = np.log(cval/co2p)
                    sign_coeff[j, ialt] = 1

                m, c, errm, errc = linear_regre_witherr(co2p, logcval)
                # if j == ialt:
                #     print(j, ialt)
                #     print(cval)
                #     print(logcval)
                #     print(m, c)

                m_coeff[j, ialt] = m
                c_coeff[j, ialt] = c
                errm_coeff[j, ialt] = errm
                errc_coeff[j, ialt] = errc
        elif ndim == 1:
            cval = np.array([co[ialt] for co in coeffs])

            if np.all(cval < 0):
                logcval = np.log(-cval/co2p)
                sign_coeff[ialt] = -1
            elif np.any(cval < 0):
                raise ValueError('Only some value is negative! at ({})'.format(ialt))
            else:
                logcval = np.log(cval/co2p)
                sign_coeff[ialt] = 1

            m, c, errm, errc = linear_regre_witherr(co2p, logcval)

            m_coeff[ialt] = m
            c_coeff[ialt] = c
            errm_coeff[ialt] = errm
            errc_coeff[ialt] = errc
        else:
            raise ValueError('Not implemented for ndim = {}'.format(ndim))

    return m_coeff, c_coeff, sign_coeff, errm_coeff, errc_coeff


def interp_coeff_logco2(coeffs, co2_profs):
    """
    Interpolates log(a/cco2)
    """

    ndim = coeffs[0].ndim
    int_fun = np.empty(coeffs[0].shape, dtype = object)

    sign_coeff = np.zeros(coeffs[0].shape)

    n_alts = coeffs[0].shape[0]

    for ialt in range(n_alts):
        co2p = np.array([co[ialt] for co in co2_profs])
        if ndim == 2:
            for j in range(n_alts):
                cval = np.array([co[j, ialt] for co in coeffs])

                if np.all(cval < 0):
                    print('All values are negative! at ({},{})\n'.format(j, ialt))
                    logcval = np.log(-cval/co2p)
                    sign_coeff[j, ialt] = -1
                elif np.any(cval < 0):
                    raise ValueError('Only some value is negative! at ({},{})'.format(j, ialt))
                else:
                    logcval = np.log(cval/co2p)
                    sign_coeff[j, ialt] = 1

                int_fun[j, ialt] = interpolate.interp1d(co2p, logcval)
        elif ndim == 1:
            cval = np.array([co[ialt] for co in coeffs])

            if np.all(cval < 0):
                logcval = np.log(-cval/co2p)
                sign_coeff[ialt] = -1
            elif np.any(cval < 0):
                raise ValueError('Only some value is negative! at ({})'.format(ialt))
            else:
                logcval = np.log(cval/co2p)
                sign_coeff[ialt] = 1

            int_fun[ialt] = interpolate.interp1d(co2p, logcval)
        else:
            raise ValueError('Not implemented for ndim = {}'.format(ndim))

    return int_fun, sign_coeff


def coeff_from_interp(int_fun, sign_coeff, co2_prof):
    """
    Reconstructs the acoeff.
    """

    coeff = np.zeros(int_fun.shape)

    n_alts = int_fun.shape[0]
    ndim = int_fun.ndim

    for ialt in range(n_alts):
        if ndim == 1:
            interplog = int_fun[ialt](co2_prof[ialt])
        else:
            interplog = np.array([intfu(co2_prof[ialt]) for intfu in int_fun[..., ialt]])

        coeff[..., ialt] = sign_coeff[..., ialt] * co2_prof[ialt] * np.exp(interplog)

    return coeff


def coeff_from_linfit(m_coeff, c_coeff, sign_coeff, co2_prof):
    """
    Reconstructs the acoeff.
    """

    coeff = np.zeros(m_coeff.shape)

    n_alts = m_coeff.shape[0]

    for ialt in range(n_alts):
        coeff[..., ialt] = sign_coeff[..., ialt] * co2_prof[ialt] * np.exp(m_coeff[..., ialt]*co2_prof[ialt] + c_coeff[..., ialt])

    return coeff


def delta_xi(xis, cco2, n_alts = 40):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """

    fu = 0.
    for atm in allatms:
        hr = all_coeffs[(atm, cco2, 'hr_ref')][:n_alts]
        hr_somma = hr_from_xi(xis, atm, cco2)
        fu += atmweigths[atm] * np.sum((hr - hr_somma)**2)

    return fu


def delta_xi_tot_fomi(xis, cco2, n_alts = 40):
    """
    Modified delta function at page 511 bottom. Gives a vector with differences for each atm profile.
    """

    fu = 0.0
    for ialt in range(n_alts):
        fuialt = delta_xi_at_x0(xis, cco2, ialt, squared_residuals = True)
        fu += np.mean(fuialt)

    return fu

def delta_xi_tot(xis, cco2, n_alts = 40):
    """
    Modified delta function at page 511 bottom. Gives a vector with differences for each atm profile.
    """
    print('Questa è sbagliata perchè elevo al quadrato anche i pesi delle diverse atmosfere, è giusta delta_xi_tot_fomi\n')

    fu = np.zeros(len(allatms))
    for ialt in range(n_alts):
        fuialt = delta_xi_at_x0(xis, cco2, ialt)
        for i, atm in enumerate(allatms):
            fu[i] += fuialt[i]**2

    fu = np.sqrt(fu)

    return fu

def delta_xi_at_x0(xis, cco2, ialt, atmweigths = atmweigths, all_coeffs = all_coeffs, atm_pt = atm_pt, squared_residuals = False):
    """
    This is done for a single altitude x0.
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """

    fu = np.zeros(len(allatms))
    for i, atm in enumerate(allatms):
        hr = all_coeffs[(atm, cco2, 'hr_ref')][ialt]
        #hr_somma = hr_from_xi(xis, atm, cco2)[ialt]
        hr_somma = hr_from_xi_at_x0(xis, atm, cco2, ialt)

        # atmweights will be squared by the loss function inside least_quares
        fu[i] = np.sqrt(atmweigths[atm]) * (hr_somma - hr)

        # if not squared_residuals:
        #     print(atm, atmweigths[atm])
        #     fu[i] = atmweigths[atm] * (hr_somma - hr)
        # else:
        #     fu[i] = atmweigths[atm] * (hr_somma - hr)**2

    return fu


def delta_xi_at_x0_afit(xis, cco2, ialt, xis_b, atmweigths = atmweigths, all_coeffs = all_coeffs, hr_ref_nam = 'hr_ref', atm_pt = atm_pt):
    """
    This is done for a single altitude x0.
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """
    #print('atmweigths: ', atmweigths)

    fu = np.zeros(len(allatms))
    for i, atm in enumerate(allatms):
        hr = all_coeffs[(atm, cco2, hr_ref_nam)][ialt]
        hr_somma = hr_from_xi_at_x0_afit(xis, atm, cco2, ialt, xis_b, all_coeffs = all_coeffs)

        # atmweights will be squared by the loss function inside least_quares
        fu[i] = np.sqrt(atmweigths[atm]) * (hr_somma - hr)

    #print(fu)
    return fu


def delta_xi_at_x0_bfit(xis, cco2, ialt, xis_a, atmweigths = atmweigths, all_coeffs = all_coeffs, hr_ref_nam = 'hr_ref', atm_pt = atm_pt):
    """
    This is done for a single altitude x0.
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """
    #print('atmweigths: ', atmweigths)

    fu = np.zeros(len(allatms))
    for i, atm in enumerate(allatms):
        hr = all_coeffs[(atm, cco2, hr_ref_nam)][ialt]
        hr_somma = hr_from_xi_at_x0_bfit(xis, atm, cco2, ialt, xis_a, all_coeffs = all_coeffs)

        # atmweights will be squared by the loss function inside least_quares
        fu[i] = np.sqrt(atmweigths[atm]) * (hr_somma - hr)

    #print(fu)
    return fu

#
# def lossfu(resi, atmweigths = atmweigths):
#     """
#     Fomi loss function.
#     """
#
#     return


def delta_xi_at_x0_tot(xis, cco2, ialt, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths, squared_residuals = True):
    """
    Modified delta function at page 511 bottom. Gives a vector with differences for each atm profile.
    """

    fu = delta_xi_at_x0(xis, cco2, ialt, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths, squared_residuals = squared_residuals)

    fu = np.sum(fu)

    return fu


def jacdelta_xi_tot(xis, cco2, n_alts = 40):
    """
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """

    J = np.empty((len(allatms), len(xis)))
    jacall = jacdelta_xi_all_x0s_fast(xis, cco2)
    delta = delta_xi_tot(xis, cco2)
    alldeltas = []
    for ialt in range(n_alts):
        alldeltas.append(delta_xi_at_x0(xis, cco2, ialt))

    for i, atm in enumerate(allatms):
        for k in range(len(xis)):
            #print(i,k)
            J[i,k] = 1/(delta[i]) * np.sum([alldeltas[ialt][i]*jacall[i,k,ialt] for ialt in range(n_alts)])

    #print(np.mean(J))
    return J


def jacdelta_xi_at_x0(xis, cco2, ialt, atmweigths = atmweigths, all_coeffs = all_coeffs, atm_pt = atm_pt):
    """
    Jacobian of delta_xi_at_x0.
    """

    J = np.empty((len(allatms), len(xis)))

    for i, atm in enumerate(allatms):
        temp = atm_pt[(allatms[i], 'temp')]
        surf_temp = atm_pt[(allatms[i], 'surf_temp')]

        for k in range(len(xis)):
            acoeff = all_coeffs[(allatms[k], cco2, 'acoeff')]
            bcoeff = all_coeffs[(allatms[k], cco2, 'bcoeff')]
            asurf = all_coeffs[(allatms[k], cco2, 'asurf')]
            bsurf = all_coeffs[(allatms[k], cco2, 'bsurf')]
            J[i,k] = atmweigths[allatms[i]]/np.sum(xis) * (hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, ialt) - hr_from_xi(xis, allatms[i], cco2)[ialt])
            #J[i,k] = atmweigths[allatms[i]] * hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, ialt)

    return J


def jacdelta_xi_at_x0_afit(xis, cco2, ialt, xis_b, atmweigths = atmweigths, all_coeffs = all_coeffs, hr_ref_nam = 'hr_ref', atm_pt = atm_pt):
    """
    Jacobian of delta_xi_at_x0_afit.
    xis_b is not used, but the code expects the same parameters that are used by delta_xi_at_x0_afit
    """

    J = np.empty((len(allatms), len(xis)))

    for i, atm in enumerate(allatms):
        temp = atm_pt[(allatms[i], 'temp')]
        surf_temp = atm_pt[(allatms[i], 'surf_temp')]

        phi_fun = np.exp(-E_fun/(kbc*temp))
        phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

        agn = coeff_from_xi_at_x0(xis, cco2, ialt, cnam = 'acoeff', all_coeffs = all_coeffs)
        agn_surf = coeff_from_xi_at_x0(xis, cco2, ialt, cnam = 'asurf', all_coeffs = all_coeffs)

        for k in range(len(xis)):
            acoeff = all_coeffs[(allatms[k], cco2, 'acoeff')]
            asurf = all_coeffs[(allatms[k], cco2, 'asurf')]

            ajac = np.sum((acoeff[:, ialt] - agn) * phi_fun) # il contributo della colonna
            # print(ajac)
            # print(len(acoeff[:, ialt]), len(agn), len(phi_fun))
            # ajac += (asurf[ialt] - agn_surf) * phi_fun_g
            # print(asurf[ialt], agn_surf, phi_fun_g)
            # print('aaaaa')
            # print(ajac, agn, acoeff[:, ialt], phi_fun)
            # print(np.sqrt(atmweigths[allatms[i]])/np.sum(xis))

            J[i,k] = np.sqrt(atmweigths[allatms[i]])/np.sum(xis) * ajac
            # sys.exit()

    return J


def jacdelta_xi_at_x0_bfit(xis, cco2, ialt, xis_a, atmweigths = atmweigths, all_coeffs = all_coeffs, hr_ref_nam = 'hr_ref', atm_pt = atm_pt):
    """
    Jacobian of delta_xi_at_x0_bfit.
    xis_a is not used, but the code expects the same parameters that are used by delta_xi_at_x0_bfit
    """

    J = np.empty((len(allatms), len(xis)))

    for i, atm in enumerate(allatms):
        temp = atm_pt[(allatms[i], 'temp')]
        surf_temp = atm_pt[(allatms[i], 'surf_temp')]

        phi_fun = np.exp(-E_fun/(kbc*temp))
        phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

        bgn = coeff_from_xi_at_x0(xis, cco2, ialt, cnam = 'bcoeff', all_coeffs = all_coeffs)
        bgn_surf = coeff_from_xi_at_x0(xis, cco2, ialt, cnam = 'bsurf', all_coeffs = all_coeffs)

        for k in range(len(xis)):
            bcoeff = all_coeffs[(allatms[k], cco2, 'bcoeff')]
            bsurf = all_coeffs[(allatms[k], cco2, 'bsurf')]

            bjac = np.sum((bcoeff[:, ialt] - bgn) * phi_fun) # il contributo della colonna
            bjac += (bsurf[ialt] - bgn_surf) * phi_fun_g

            J[i,k] = np.sqrt(atmweigths[allatms[i]])/np.sum(xis) * phi_fun[ialt] * bjac

    return J


def jacdelta_xi_all_x0s_fast(xis, cco2, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths):
    """
    Jacobian of delta_xi_at_x0.
    """

    J = np.empty((len(allatms), len(xis), n_alts))

    for i, atm in enumerate(allatms):
        temp = atm_pt[(allatms[i], 'temp')]
        surf_temp = atm_pt[(allatms[i], 'surf_temp')]

        hrsomma = hr_from_xi(xis, allatms[i], cco2)
        for k in range(len(xis)):
            acoeff = all_coeffs[(allatms[k], cco2, 'acoeff')]
            bcoeff = all_coeffs[(allatms[k], cco2, 'bcoeff')]
            asurf = all_coeffs[(allatms[k], cco2, 'asurf')]
            bsurf = all_coeffs[(allatms[k], cco2, 'bsurf')]
            hrsing = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)
            for ialt in range(n_alts):
                J[i,k,ialt] = atmweigths[allatms[i]]/np.sum(xis) * (hrsing[ialt] - hrsomma[ialt])

    return J


###########################################################
# Upper trans region

def transrecformula(alpha, L_esc, lamb, eps125, co2vmr, MM, temp, n_trans = 7):
    """
    Recurrence formula in the upper transition region (with alpha).

    n_trans = n_alts_trhi-n_alts_trlo+1
    """
    eps125 = eps125 * cp / (24*60*60)

    phi_fun = np.exp(-E_fun/(kbc*temp))
    dj = L_esc*alpha

    eps_gn = np.zeros(n_trans)
    eps_gn[0] = 1.10036e-10*eps125/(co2vmr[0] * (1-lamb[0]))

    for j in range(1, n_trans): # Formula 9
        Djj = 0.25*(dj[j-1] + 3*dj[j])
        Djjm1 = 0.25*(dj[j] + 3*dj[j-1])

        Fj = (1 - lamb[j]*(1-Djj))
        Fjm1 = (1 - lamb[j-1]*(1-Djjm1))
        eps_gn[j] = (Fjm1*eps_gn[j-1] + Djjm1*phi_fun[j-1] - Djj*phi_fun[j])/Fj

    #MM = np.ones(len(alts)) * (0.79*28+0.21*32) # Molecular mass
    fac = (2.63187e11 * co2vmr * (1-lamb))/MM
    eps = fac * eps_gn # Formula 7

    eps = eps * (24*60*60) / cp # convert back to K/day

    return eps[1:]


def delta_alpha_rec(alpha, cco2, cose_upper_atm, n_alts_trlo = 50, n_alts_trhi = 56, weigths = np.ones(len(allatms)), all_coeffs = None, atm_pt = atm_pt):
    """
    This is done for all n_trans = 6 altitudes at a time.
    """

    n_trans = n_alts_trhi-n_alts_trlo+1

    fu = []
    for i, atm in enumerate(allatms):
        hr_ref = all_coeffs[(atm, cco2, 'hr_nlte')][n_alts_trlo:n_alts_trhi]

        L_esc = cose_upper_atm[(atm, cco2, 'L_esc')][n_alts_trlo-1:n_alts_trhi]
        lamb = cose_upper_atm[(atm, cco2, 'lamb')][n_alts_trlo-1:n_alts_trhi]
        co2vmr = cose_upper_atm[(atm, cco2, 'co2vmr')][n_alts_trlo-1:n_alts_trhi]
        MM = cose_upper_atm[(atm, cco2, 'MM')][n_alts_trlo-1:n_alts_trhi]
        temp = atm_pt[(atm, 'temp')][n_alts_trlo-1:n_alts_trhi]
        eps125 = cose_upper_atm[(atm, cco2, 'eps125')]

        hr_calc = transrecformula(alpha, L_esc, lamb, eps125, co2vmr, MM, temp, n_trans = n_alts_trhi-n_alts_trlo+1)

        # atmweights will be squared by the loss function inside least_quares
        fu.append(hr_calc - hr_ref)

    fu = np.stack(fu)
    fu = weigths[:, np.newaxis]*fu**2
    fu = np.sqrt(np.sum(fu, axis = 0)) # in questo modo fu ha dimensione n_trans
    #resid = np.sqrt(atmweigths[i] * np.sum((hr_calc - hr)**2))

    return fu


def delta_alpha_rec2(alpha, cco2, cose_upper_atm, n_alts_trlo = 50, n_alts_trhi = 56, weigths = np.ones(len(allatms)), all_coeffs = None, atm_pt = atm_pt):
    """
    This is done for all n_trans = 6 altitudes at a time.
    """

    n_trans = n_alts_trhi-n_alts_trlo+1

    fu = []
    for i, atm in enumerate(allatms):
        hr_ref = all_coeffs[(atm, cco2, 'hr_nlte')][n_alts_trlo:n_alts_trhi]

        L_esc = cose_upper_atm[(atm, cco2, 'L_esc')][n_alts_trlo-1:n_alts_trhi]
        lamb = cose_upper_atm[(atm, cco2, 'lamb')][n_alts_trlo-1:n_alts_trhi]
        co2vmr = cose_upper_atm[(atm, cco2, 'co2vmr')][n_alts_trlo-1:n_alts_trhi]
        MM = cose_upper_atm[(atm, cco2, 'MM')][n_alts_trlo-1:n_alts_trhi]
        temp = atm_pt[(atm, 'temp')][n_alts_trlo-1:n_alts_trhi]
        eps125 = cose_upper_atm[(atm, cco2, 'eps125')]

        hr_calc = transrecformula(alpha, L_esc, lamb, eps125, co2vmr, MM, temp, n_trans = n_alts_trhi-n_alts_trlo+1)

        # atmweights will be squared by the loss function inside least_quares
        fu.append(hr_calc - hr_ref)

    fu = np.concatenate(fu)
    # fu = weigths[:, np.newaxis]*fu**2
    # fu = np.sqrt(np.sum(fu, axis = 0)) # in questo modo fu ha dimensione n_trans
    # #resid = np.sqrt(atmweigths[i] * np.sum((hr_calc - hr)**2))

    return fu


def recformula(alpha, L_esc, lamb, hr, co2vmr, MM, temp, n_alts_trlo = 50, n_alts_trhi = 56):
    """
    Recurrence formula in the upper transition region (with alpha).

    With full vectors.
    """
    n_alts = len(hr)
    phi_fun = np.exp(-E_fun/(kbc*temp))

    eps125 = hr[n_alts_trlo-1] * cp / (24*60*60)

    alpha_ok = np.ones(n_alts)
    alpha_ok[n_alts_trlo-1:n_alts_trhi] = alpha
    dj = L_esc*alpha_ok

    eps_gn = np.zeros(n_alts)
    eps_gn[n_alts_trlo-1] = 1.10036e-10*eps125/(co2vmr[n_alts_trlo-1] * (1-lamb[n_alts_trlo-1]))

    for j in range(n_alts_trlo, n_alts): # Formula 9
        Djj = 0.25*(dj[j-1] + 3*dj[j])
        Djjm1 = 0.25*(dj[j] + 3*dj[j-1])

        Fj = (1 - lamb[j]*(1-Djj))
        Fjm1 = (1 - lamb[j-1]*(1-Djjm1))
        eps_gn[j] = (Fjm1*eps_gn[j-1] + Djjm1*phi_fun[j-1] - Djj*phi_fun[j])/Fj

    fac = (2.63187e11 * co2vmr * (1-lamb))/MM
    hr[n_alts_trlo:] = fac[n_alts_trlo:] * eps_gn[n_alts_trlo:]  # Formula 7
    hr[n_alts_trlo:] = hr[n_alts_trlo:] * (24*60*60) / cp # convert back to K/day

    return hr


###########################################################

def plot_pdfpages(filename, figs, save_single_figs = False, fig_names = None):
    """
    Saves a list of figures to a pdf file.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages(filename)
    for fig in figs:
        pdf.savefig(fig)
    pdf.close()

    if save_single_figs:
        indp = filename.index('.')
        cartnam = filename[:indp]+'_figures/'
        if not os.path.exists(cartnam):
            os.mkdir(cartnam)
        if fig_names is None:
            fig_names = ['pag_{}'.format(i+1) for i in range(len(figs))]
        for fig,nam in zip(figs, fig_names):
            fig.savefig(cartnam+nam+'.pdf')

    return


def manuel_plot(y, xs, labels, xlabel = None, ylabel = None, title = None, xlimdiff = None, colors = None, linestyles = None, xlim = (None, None), ylim = (None, None), orizlines = [70., 85.]):
    """
    Plots plt.plot(x, y, lab) for each x in xs. Plots the differences of all xs wrt xs[0] in a side plot.
    """
    fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
    if colors is None: colors = color_set(len(xs))
    if linestyles is None: linestyles = len(xs)*['-']
    i = 0
    for x, lab, col, lst in zip(xs, labels, colors, linestyles):
        a0.plot(x, y, label = lab, color = col, linestyle = lst)
        if i == 0:
            i+=1
            continue
        if i == 1:
            a1.axvline(0., color = 'grey', alpha = 0.6)
            a1.axvline(0.5, color = 'grey', alpha = 0.4, linestyle = ':', linewidth = 0.8)
            a1.axvline(-0.5, color = 'grey', alpha = 0.4, linestyle = ':', linewidth = 0.8)
            a1.axvline(1.0, color = 'grey', alpha = 0.4, linestyle = '--', linewidth = 0.8)
            a1.axvline(-1.0, color = 'grey', alpha = 0.4, linestyle = '--', linewidth = 0.8)
        a1.plot(x - xs[0], y, color = col, linestyle = lst)
        i+=1

    for orizli, col in zip(orizlines, ['red', 'orange', 'green', 'blue']):
        a0.axhline(orizli, color = col, alpha = 0.6, linestyle = '--')
        a1.axhline(orizli, color = col, alpha = 0.6, linestyle = '--')
    a0.grid()
    a1.grid()
    if xlimdiff is not None:
        a1.set_xlim(xlimdiff)
    a0.legend(loc = 3)

    a0.set_xlim(xlim)
    a0.set_ylim(ylim)
    a1.set_ylim(ylim)

    if xlabel is not None: a0.set_xlabel(xlabel)
    if ylabel is not None: a0.set_ylabel(ylabel)
    if title is not None: a0.set_title(title)

    #fig.tight_layout()

    return fig, a0, a1


def adjust_ax_scale(axes, sel_axis = 'both'):
    """
    Given a set of axes, uniformizes the scales.
    < sel_axis > : 'x', 'y' or 'both' (default)
    """

    if sel_axis == 'x' or sel_axis == 'both':
        limits_min = []
        limits_max = []
        for ax in axes:
            limits_min.append(ax.get_xlim()[0])
            limits_max.append(ax.get_xlim()[1])

        maxlim = np.max(limits_max)
        minlim = np.min(limits_min)
        for ax in axes:
            ax.set_xlim((minlim,maxlim))

    if sel_axis == 'y' or sel_axis == 'both':
        limits_min = []
        limits_max = []
        for ax in axes:
            limits_min.append(ax.get_ylim()[0])
            limits_max.append(ax.get_ylim()[1])

        maxlim = np.max(limits_max)
        minlim = np.min(limits_min)
        for ax in axes:
            ax.set_ylim((minlim,maxlim))

    return


def color_set(n, cmap = 'nipy_spectral', full_cb_range = False):
    """
    Gives a set of n well chosen (hopefully) colors, darker than bright_thres. bright_thres ranges from 0 (darker) to 1 (brighter).

    < full_cb_range > : if True, takes all cb values. If false takes the portion 0.05/0.95.
    """
    cmappa = cm.get_cmap(cmap)
    colors = []

    if full_cb_range:
        valori = np.linspace(0.0,1.0,n)
    else:
        valori = np.linspace(0.05,0.95,n)

    for cos in valori:
        colors.append(cmappa(cos))

    return colors
