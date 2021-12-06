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
from scipy.interpolate import interp1d

from subprocess import call

import pickle

if os.uname()[1] == 'xaru':
    sys.path.insert(0, '/home/fedef/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fedef/Research/git/pythall/')
    cart_out = '/home/fedef/Research/lavori/CO2_cooling/new_param/LTE/'
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
kb = 1.38065e-19 # boltzmann constant to use with P in hPa, n in cm-3, T in K (10^-4 * Kb standard)

E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition

cp_0 = 1.005e7 # specific enthalpy dry air - erg g-1 K-1

NLTE_DEBUG = True
#############################################################


allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweigths = dict(zip(allatms, atmweigths))
allco2 = np.arange(1,8)

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v2.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt_v2.p'))
n_alts = 40

regrcoefpath = cart_out + '../NLTE_reparam/regrcoef_v3.p'
regrcoef = pickle.load(open(regrcoefpath, 'rb'))

alpha_cose = pickle.load(open(cart_out + '../NLTE_reparam/alpha_fit_4e.p', 'rb'))
alpha_cose['min'] = np.min(np.stack([alpha_cose[('min', i)] for i in range(1,8)]), axis = 0)
alpha_cose['max'] = np.max(np.stack([alpha_cose[('max', i)] for i in range(1,8)]), axis = 0)

nlte_corr = pickle.load(open(cart_out + '../NLTE_reparam/nlte_corr_low.p', 'rb'))

from scipy.optimize import Bounds, minimize, least_squares

#############################################################


def new_param_full_old(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, coeffs = None, coeff_file = cart_out + '../newpar_allatm/coeffs_finale.p', interp_coeffs = None):
    """
    New param valid for the full atmosphere.
    """
    n_alts_trlo = 50
    n_top = 61 # n_alts_trhi + 5

    ##### Interpolate all profiles to param grid.
    print('I am not interpolating yet! profiles should already be given on a fix grid')

    if coeffs is None:
        coeffs = pickle.load(open(coeff_file, 'rb'))

    ### Interpolation of the coefficients to the actual CO2 profile
    if interp_coeffs is None:
        interp_coeffs = precalc_interp_old(coeffs = coeffs, coeff_file = coeff_file)

    print('Coeffs from interpolation!')
    calc_coeffs = dict()
    for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf', 'alpha', 'Lesc']:
        int_fun = interp_coeffs[(nam, 'int_fun')]
        sc = interp_coeffs[(nam, 'signc')]

        coeff = coeff_from_interp_log(int_fun, sc, co2vmr)
        calc_coeffs[nam] = coeff

    hr_calc = hr_from_ab(calc_coeffs['acoeff'], calc_coeffs['bcoeff'], calc_coeffs['asurf'], calc_coeffs['bsurf'], temp, surf_temp)

    lamb = calc_lamb(pres, temp, ovmr, o2vmr, n2vmr)
    MM = calc_MM(ovmr, o2vmr, n2vmr)
    #alpha_ = 10.*np.ones(n_alts_trhi-n_alts_trlo+1)
    hr_calc = recformula(calc_coeffs['alpha'], calc_coeffs['Lesc'], lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = n_alts_trlo, n_alts_trhi = n_top)

    ##### HERE the cool-to-space part
    # now for the cs region:
    # Phi_165 = eps_gn[n_alts_cs] + phi_fun[n_alts_cs]
    # eps[n_alts_cs:] = fac[n_alts_cs:] * (Phi_165 - phi_fun[j])

    return hr_calc

def precalc_interp_old(coeffs = None, coeff_file = cart_out + '../newpar_allatm/coeffs_finale.p'):

    if coeffs is None:
        coeffs = pickle.load(open(coeff_file, 'rb'))

    interp_coeffs = dict()
    for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf', 'alpha', 'Lesc']:
        int_fun, signc = interp_coeff_logco2(coeffs[nam], coeffs['co2profs'])
        interp_coeffs[(nam, 'int_fun')] = int_fun
        interp_coeffs[(nam, 'signc')] = signc

    return interp_coeffs

def precalc_interp(coeffs = None, coeff_file = cart_out + '../reparam_allatm/coeffs_finale.p'):
    """
    Calculates the interpolating functions. (this makes new_param_full much faster)
    """
    alt1 = 40
    alt2 = 51
    n_top = 65

    if coeffs is None:
        coeffs = pickle.load(open(coeff_file, 'rb'))

    ########################################################
    #### PREPARING INTERPOLATION FUNCTIONS (should be done only once in a climate run)

    co2profs = coeffs['co2profs']

    interp_coeffs = dict() ## l'interpolazione
    for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        kosi = ['c1', 'm1', 'm2']
        if 'surf' in nam:
            kosi = ['c', 'm']

        for regco in kosi:
            ko = coeffs[(nam, regco)]
            if regco in ['c', 'c1']:
                int_fun, signc = interp_coeff_logco2(ko, co2profs)
                interp_coeffs[(nam, regco, 'int_fun')] = int_fun
                interp_coeffs[(nam, regco, 'signc')] = signc
            else:
                int_fun = interp_coeff_linco2(ko, co2profs)
                interp_coeffs[(nam, regco, 'int_fun')] = int_fun

    #### NLTE correction (low trans)
    nam = 'nltecorr'
    for regco in ['c', 'm1', 'm2', 'm3', 'm4']:
        ko = coeffs[(nam, regco)]
        int_fun = interp_coeff_linco2(ko, co2profs[:, alt1:alt2])
        interp_coeffs[(nam, regco, 'int_fun')] = int_fun

    alphas_all = coeffs['alpha']
    intfutu = []
    for go in range(alphas_all.shape[-1]):
        int_fun = interp_coeff_linco2(alphas_all[..., go], co2profs[:, alt2:n_top+1])
        intfutu.append(int_fun)
    interp_coeffs[('alpha', 'int_fun')] = intfutu

    interp_coeffs[('alpha_min', 'int_fun')] = interp_coeff_linco2(coeffs['alpha_min'], co2profs[:, alt2:n_top+1])
    interp_coeffs[('alpha_max', 'int_fun')] = interp_coeff_linco2(coeffs['alpha_max'], co2profs[:, alt2:n_top+1])

    Lesc_all = coeffs['Lesc']
    int_fun = interp_coeff_linco2(Lesc_all, co2profs)
    interp_coeffs[('Lesc', 'int_fun')] = int_fun

    interp_coeffs['uco2'] = coeffs['uco2']
    interp_coeffs['alts'] = coeffs['alts']

    ###### END PREPARING
    ##########################################################

    return interp_coeffs


def new_param_full_allgrids(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, coeffs = None, coeff_file = cart_out + '../reparam_allatm/coeffs_finale.p', interp_coeffs = None, debug_Lesc = None):
    """
    Wrapper for new_param_full that takes in input vectors on arbitrary grids.
    """

    if interp_coeffs is None:
        print('Precalculate interp function for faster calculations')
        interp_coeffs = precalc_interp(coeffs = coeffs, coeff_file = coeff_file)

    ## custom x grid
    x = np.log(1000./pres)

    ## reference x grid
    x_ref_max = 16.735
    if np.max(x) > x_ref_max:
        x_ref = np.arange(0.125, np.max(x), 0.25)
    else:
        x_ref = np.arange(0.125, x_ref_max + 0.001, 0.25)

    ##### INTERPOLATE EVERYTHING TO REFERENCE GRID HERE ####
    spl = spline(x, temp)
    temp_rg = spl(x_ref)

    spl = spline(x, np.log(pres))
    pres_rg = spl(x_ref)
    pres_rg = np.exp(pres_rg)

    spl = spline(x, ovmr)
    ovmr_rg = spl(x_ref)

    spl = spline(x, o2vmr)
    o2vmr_rg = spl(x_ref)

    spl = spline(x, co2vmr)
    co2vmr_rg = spl(x_ref)

    spl = spline(x, n2vmr)
    n2vmr_rg = spl(x_ref)

    ########## Call new param

    hr_calc_fin = new_param_full(temp_rg, surf_temp, pres_rg, co2vmr_rg, ovmr_rg, o2vmr_rg, n2vmr_rg, coeffs = coeffs, coeff_file = coeff_file, interp_coeffs = interp_coeffs, debug_Lesc = debug_Lesc)

    ##### INTERPOLATE OUTPUT TO ORIGINAL GRID ####

    spl = spline(x_ref, hr_calc_fin)
    hr_calc = spl(x)

    return hr_calc


def new_param_full(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, coeffs = None, coeff_file = cart_out + '../reparam_allatm/coeffs_finale.p', interp_coeffs = None, debug_Lesc = None):
    """
    New param with new strategy (1/10/21).
    """

    alt1 = 40 # start non-LTE correction
    alt2 = 51 # start non-LTE upper region (alpha, L_esc)
    n_top = 65 # max alt to apply alpha correction

    if interp_coeffs is None:
        print('Precalculate interp function for faster calculations')
        interp_coeffs = precalc_interp(coeffs = coeffs, coeff_file = coeff_file)

    alts = interp_coeffs['alts']

    ############################################
    ##### INTERPOLATING TO ACTUAL CO2
    calc_coeffs = dict()
    for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf', 'nltecorr']:
        for regco in ['c', 'm', 'c1', 'm1', 'm2', 'm3', 'm4']:
            if (nam, regco, 'int_fun') not in interp_coeffs:
                continue
            int_fun = interp_coeffs[(nam, regco, 'int_fun')]

            if regco in ['c', 'c1'] and nam != 'nltecorr':
                sc = interp_coeffs[(nam, regco, 'signc')]
                coeff = coeff_from_interp_log(int_fun, sc, co2vmr)
            else:
                coeff = coeff_from_interp_lin(int_fun, co2vmr)

            calc_coeffs[(nam, regco)] = coeff

    # upper atm
    intfutu = interp_coeffs[('alpha', 'int_fun')]
    allco = []
    for intfu in intfutu:
        allco.append(coeff_from_interp_lin(intfu, co2vmr[alt2:n_top+1]))
    calc_coeffs['alpha_fit'] = np.stack(allco).T

    calc_coeffs['alpha_min'] = coeff_from_interp_lin(interp_coeffs[('alpha_min', 'int_fun')], co2vmr[alt2:n_top+1])
    calc_coeffs['alpha_max'] = coeff_from_interp_lin(interp_coeffs[('alpha_max', 'int_fun')], co2vmr[alt2:n_top+1])

    L_all = coeff_from_interp_lin(interp_coeffs[('Lesc', 'int_fun')], co2vmr)
    uco2 = interp_coeffs['uco2']

    Lspl_all = spline(uco2, L_all, extrapolate = False)
    #uok2 = calc_co2column(alts, pres, temp, co2vmr)
    MM = calc_MM(ovmr, o2vmr, n2vmr)
    uok2 = calc_co2column_P(pres, co2vmr, MM)
    L_esc = Lspl_all(uok2)
    L_esc[np.isnan(L_esc)] = 0.

    # print('! TO be changed, L_esc to be calc from L function and integrated CO2 prof above')
    # L_esc = coeff_from_interp_lin(interp_coeffs[('Lesc', 'int_fun')], co2vmr)
    if debug_Lesc is not None:
        print('Getting L_esc externally for DEBUG!!!')
        L_esc = debug_Lesc

    ###############################################
    #### END INTERP

    ###############################################
    #### CALCULATION

    #lte
    acoeff, bcoeff, asurf, bsurf = coeffs_from_eofreg_single(temp, surf_temp, calc_coeffs)
    hr_lte = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = 66)

    #### nltecorr
    hr_nlte_corr = nltecorr_from_eofreg_single(temp, surf_temp, calc_coeffs, alt1 = alt1, alt2 = alt2)

    hr_calc = hr_lte.copy()
    hr_calc[alt1:alt2] += hr_nlte_corr

    #### upper atm
    lamb = calc_lamb(pres, temp, ovmr, o2vmr, n2vmr)
    alpha = alpha_from_fit(temp, surf_temp, lamb, calc_coeffs['alpha_fit'], alpha_max = calc_coeffs['alpha_max'], alpha_min = calc_coeffs['alpha_min'])

    hr_calc_fin = recformula(alpha, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, n_alts_cs = n_top, ovmr = ovmr)

    ##### HERE the cool-to-space part
    # now for the cs region:
    # Phi_165 = eps_gn[n_alts_cs] + phi_fun[n_alts_cs]
    # eps[n_alts_cs:] = fac[n_alts_cs:] * (Phi_165 - phi_fun[j])

    if debug_Lesc:
        DEBUG = [alpha, MM, lamb, L_esc, hr_calc, hr_lte]

        return hr_calc_fin, DEBUG
    else:
        return hr_calc_fin


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

        coeffs.append(coeff_from_interp_log(int_fun, sc, co2pr))

    acoeff, bcoeff, asurf, bsurf = coeffs

    hr_calc = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)

    return hr_calc


def old_param(alts, temp, pres, CO2prof, Oprof = None, O2prof = None, N2prof = None, cart_run_fomi = '/home/fabiano/Research/lavori/CO2_cooling/cart_run_fomi/', input_in_ppm = False):
    """
    Run the old param.

    Default input in ppm. Instead set input_in_ppm to False to specify concentrations.

    """
    if input_in_ppm:
        fact = 1.0
    else:
        print('Converting inputs to ppm!')
        fact = 1.e6

    fil_VMR = cart_run_fomi + 'gases_120.dat'
    alt_manuel, mol_vmrs, molist, molnums = sbm.read_input_vmr_man(fil_VMR, version = 2)

    splCO2 = spline(alts, CO2prof)
    CO2con = splCO2(alt_manuel)
    print('piooo CO2', np.median(CO2con))
    mol_vmrs['CO2'] = CO2con*fact
    # Also update O2, O, N2
    if Oprof is not None:
        splO = spline(alts, Oprof)
        Ocon = splO(alt_manuel)
        print('piooo O', np.median(Ocon))
        mol_vmrs['O'] = Ocon*fact
    if O2prof is not None:
        splO2 = spline(alts, O2prof)
        O2con = splO2(alt_manuel)
        print('piooo O2', np.median(O2con))
        mol_vmrs['O2'] = O2con*fact
    if N2prof is not None:
        splN2 = spline(alts, N2prof)
        N2con = splN2(alt_manuel)
        print('piooo N2', np.median(N2con))
        mol_vmrs['N2'] = N2con*fact

    splT = spline(alts, temp)
    temp = splT(alt_manuel)

    splP = spline(alts,np.log(pres))
    pres = splP(alt_manuel)
    pres = np.exp(pres)

    filename = cart_run_fomi + 'atm_manuel.dat'
    sbm.scriviinputmanuel(alt_manuel, temp, pres, filename)

    filename = cart_run_fomi + 'vmr_atm_manuel.dat'
    sbm.write_input_vmr_man(filename, alt_manuel, mol_vmrs, hit_gas_list = molist, hit_gas_num = molnums, version = 2)

    wd = os.getcwd()
    os.chdir(cart_run_fomi)
    call('./fomi_mipas')
    os.chdir(wd)
    nomeout = cart_run_fomi + 'output__mipas.dat'
    alt_fomi, x_fomi, cr_fomi = leggioutfomi(nomeout)

    return alt_fomi, x_fomi, cr_fomi


def trova_spip(ifile, hasha = '#', read_past = False):
    """
    Trova il '#' nei file .dat
    """
    gigi = 'a'
    while gigi != hasha :
        linea = ifile.readline()
        gigi = linea[0]
    else:
        if read_past:
            return linea[1:]
        else:
            return


def leggioutfomi(nomeout):
    """
    Reads Fomichev output.
    :param nomeout:
    :return:
    """
    fi = open(nomeout,'r')
    trova_spip(fi)

    data = np.array([map(float, line.split()) for line in fi])

    alt_fomi = np.array(data[:,0])
    x_fomi = np.array(data[:,4])
    cr_fomi = np.array(data[:,5])

    return alt_fomi, x_fomi, cr_fomi


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
    n_alts = np.min([len(temp), acoeff.shape[1]])

    epsilon_ab_tot = np.zeros(n_alts, dtype = float)

    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    for xi in range(n_alts):
        epsilon_ab_tot[xi] = np.sum((acoeff[:max_alts, xi] + bcoeff[:max_alts, xi]* phi_fun[xi]) * phi_fun[:max_alts]) # il contributo della colonna
        epsilon_ab_tot[xi] += (asurf[xi] + bsurf[xi]* phi_fun[xi]) * phi_fun_g

    return epsilon_ab_tot


def hr_from_ab_decomposed(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = 51):
    """
    This is the LTE cooling rate given a certain set of a and b coefficients, separating the contribution from a and b coeffs.
    """
    n_alts = np.min([len(temp), acoeff.shape[1]])

    epsilon_ab_tot_a = np.zeros(n_alts, dtype = float)
    epsilon_ab_tot_b = np.zeros(n_alts, dtype = float)

    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    for xi in range(n_alts):
        epsilon_ab_tot_a[xi] = np.sum(acoeff[:max_alts, xi] * phi_fun[:max_alts]) # il contributo della colonna
        epsilon_ab_tot_a[xi] += asurf[xi] * phi_fun_g

        epsilon_ab_tot_b[xi] = np.sum(bcoeff[:max_alts, xi]* phi_fun[xi] * phi_fun[:max_alts]) # il contributo della colonna
        epsilon_ab_tot_b[xi] += bsurf[xi]* phi_fun[xi] * phi_fun_g

    return epsilon_ab_tot_a, epsilon_ab_tot_b


def hr_from_ab_diagnondiag(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = 51):
    """
    This is the LTE cooling rate given a certain set of a and b coefficients, separating the contribution from a and b coeffs.
    """
    n_alts = np.min([len(temp), acoeff.shape[1]])

    epsilon_ab_tot_diag = np.zeros(n_alts, dtype = float)
    epsilon_ab_tot_nondiag = np.zeros(n_alts, dtype = float)

    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    for xi in range(n_alts):
        epsilon_ab_tot_diag[xi] = acoeff[xi, xi] * phi_fun[xi] + bcoeff[xi, xi]* phi_fun[xi]**2

        epsilon_ab_tot_nondiag[xi] = np.sum((acoeff[:max_alts, xi] + bcoeff[:max_alts, xi]* phi_fun[xi]) * phi_fun[:max_alts]) - epsilon_ab_tot_diag[xi]
        epsilon_ab_tot_nondiag[xi] += (asurf[xi] + bsurf[xi]* phi_fun[xi]) * phi_fun_g

    return epsilon_ab_tot_diag, epsilon_ab_tot_nondiag


def running_mean(var, wnd, remove_nans = False, keep_length = False):
    """
    Performs a running mean (if multidim, the mean is done on the first axis).

    < wnd > : is the window length.
    """

    rollpi_temp = []
    for i in range(len(var)):
        if i-wnd//2 < 0 or i + wnd//2 > len(var)-1:
            if remove_nans: continue
            rollpi_temp.append(np.nan*np.ones(var[0].shape))
        else:
            rollpi_temp.append(np.mean(var[i-wnd//2:i+wnd//2+1, ...], axis = 0))

    rollpi_temp = np.stack(rollpi_temp)

    return rollpi_temp


def linear_regre_witherr(x, y):
    """
    Makes a linear regression of dataset y in function of x using numpy.polyfit. Returns the coefficient m and c: y = mx + c. And their estimated error.
    """

    if type(x) is list:
        x = np.array(x)
        y = np.array(y)

    xord = np.argsort(x)
    x = x[xord]
    y = y[xord]

    res = np.polyfit(x, y, deg = 1, cov = True)
    m,c = res[0]
    covmat = res[1]

    err_m = np.sqrt(covmat[0,0])
    err_c = np.sqrt(covmat[1,1])

    return m, c, err_m, err_c


def linearregre_coeff(x, coeffs):
    trendmat = np.empty_like(coeffs[0])
    errtrendmat = np.empty_like(coeffs[0])
    cmat = np.empty_like(coeffs[0])
    errcmat = np.empty_like(coeffs[0])

    if trendmat.ndim == 2:
        for i in np.arange(trendmat.shape[0]):
            for j in np.arange(trendmat.shape[1]):
                m, c, err_m, err_c = linear_regre_witherr(x, coeffs[:,i,j])
                #coeffs, covmat = np.polyfit(years, var_set[i,j], deg = deg, cov = True)
                trendmat[i,j] = m
                errtrendmat[i,j] = err_m
                cmat[i,j] = c
                errcmat[i,j] = err_c
    elif trendmat.ndim == 1:
        for i in np.arange(trendmat.shape[0]):
            m, c, err_m, err_c = linear_regre_witherr(x, coeffs[:,i])
            #coeffs, covmat = np.polyfit(years, var_set[i,j], deg = deg, cov = True)
            trendmat[i] = m
            errtrendmat[i] = err_m
            cmat[i] = c
            errcmat[i] = err_c

    return cmat, trendmat, errcmat, errtrendmat


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


def coeffs_from_eofreg(cco2, temp, surf_temp, method = '2eof', regrcoef = regrcoef):
    """
    Reconstructs the a and b coeffs for the required atmosphere.
    """

    surfanom = surf_temp - regrcoef['surfmean']
    atm_anom_mean = regrcoef['amean']
    eof0 = regrcoef['eof0']
    eof1 = regrcoef['eof1']
    n_alts = len(eof0)

    pc0 = np.dot(temp[:n_alts]-atm_anom_mean, eof0)
    pc1 = np.dot(temp[:n_alts]-atm_anom_mean, eof1)

    coeffs = dict()
    for conam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        if 'surf' in conam:
            coeffs[conam] = regrcoef[(cco2, conam, 'c')] + regrcoef[(cco2, conam, 'm')]*surfanom
        else:
            if method == '2eof':
                coeffs[conam] = regrcoef[(cco2, conam, 'c1')] + regrcoef[(cco2, conam, 'm1')]*pc0 + regrcoef[(cco2, conam, 'm2')]*pc1
            elif method == '1eof':
                coeffs[conam] = regrcoef[(cco2, conam, 'c')] + regrcoef[(cco2, conam, 'm')]*pc0

    return coeffs['acoeff'], coeffs['bcoeff'], coeffs['asurf'], coeffs['bsurf']


def coeffs_from_eofreg_single(temp, surf_temp, singlecoef, regrcoef = regrcoef):
    """
    Reconstructs the a and b coeffs for the required atmosphere.
    """

    surfanom = surf_temp - regrcoef['surfmean']
    atm_anom_mean = regrcoef['amean']
    eof0 = regrcoef['eof0']
    eof1 = regrcoef['eof1']
    n_alts = len(eof0)

    pc0 = np.dot(temp[:n_alts]-atm_anom_mean, eof0)
    pc1 = np.dot(temp[:n_alts]-atm_anom_mean, eof1)

    coeffs = dict()
    for conam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        if 'surf' in conam:
            coeffs[conam] = singlecoef[(conam, 'c')] + singlecoef[(conam, 'm')]*surfanom
        else:
            coeffs[conam] = singlecoef[(conam, 'c1')] + singlecoef[(conam, 'm1')]*pc0 + singlecoef[(conam, 'm2')]*pc1

    return coeffs['acoeff'], coeffs['bcoeff'], coeffs['asurf'], coeffs['bsurf']


def nltecorr_from_eofreg_single(temp, surf_temp, singlecoef, regrcoef = regrcoef, alt1 = 40, alt2 = 51):
    """
    Reconstructs the a and b coeffs for the required atmosphere.
    """

    surfanom = surf_temp - regrcoef['surfmean']
    atm_anom_mean = regrcoef['amean']
    eof0 = regrcoef['eof0']
    eof1 = regrcoef['eof1']
    n_alts = len(eof0)

    pc0 = np.dot(temp[:n_alts]-atm_anom_mean, eof0)
    pc1 = np.dot(temp[:n_alts]-atm_anom_mean, eof1)

    acoeff, bcoeff, asurf, bsurf = coeffs_from_eofreg_single(temp, surf_temp, singlecoef)

    hra, hrb = hr_from_ab_diagnondiag(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts=66)

    hr_nlte_corr = singlecoef[('nltecorr', 'c')] + singlecoef[('nltecorr', 'm1')] * hra[alt1:alt2] + singlecoef[('nltecorr', 'm2')] * hrb[alt1:alt2] + singlecoef[('nltecorr', 'm3')] * pc0 + singlecoef[('nltecorr', 'm4')] * pc1

    return hr_nlte_corr


def alpha_from_fit(temp, surf_temp, lamb, alpha_fit, alpha_cose = alpha_cose, alpha_min = None, alpha_max = None, method = 'nl0', alt2 = 51, n_top = 65):
    """
    Reconstructs alpha. Method: 4e, nl0
    """

    # population upper state
    phifunz = np.exp(-E_fun/(kbc*temp[alt2:n_top+1]))
    lambdivA = lamb[alt2:n_top+1]/1.5988
    popup = lambdivA*phifunz

    popup_mean = alpha_cose['popup_mean']
    eofs_all = [alpha_cose['eof{}'.format(i)] for i in range(4)]

    dotprods = np.array([np.dot(popup-popup_mean, eoff) for eoff in eofs_all])
    dotprods2 = np.array([dotprods[0], dotprods[0]**2] + [dotprods[1], dotprods[1]**2])

    if method == '4e':
        alpha = alpha_fit[:, 0] + np.sum(alpha_fit[:, 1:] * dotprods[np.newaxis, :], axis = 1)
    elif method == 'nl0':
        alpha = alpha_fit[:, 0] + np.sum(alpha_fit[:, 1:] * dotprods2[np.newaxis, :], axis = 1)

    #print('setting constraint on alpha! check this part')
    # alpha_min e max are profiles, setting stupid numbers for now
    # if alpha_min is None: alpha_min = alpha_cose['min']
    # if alpha_max is None: alpha_max = alpha_cose['max']
    lower = alpha < alpha_min
    alpha[lower] = alpha_min[lower]
    higher = alpha > alpha_max
    alpha[higher] = alpha_max[higher]

    return alpha


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
                    #print('All values are negative! at ({},{})\n'.format(j, ialt))
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
                    #print('All values are negative! at ({},{})\n'.format(j, ialt))
                    logcval = np.log(-cval/co2p)
                    sign_coeff[j, ialt] = -1
                elif np.any(cval < 0):
                    raise ValueError('Only some value is negative! at ({},{})'.format(j, ialt))
                else:
                    logcval = np.log(cval/co2p)
                    sign_coeff[j, ialt] = 1

                int_fun[j, ialt] = interpolate.interp1d(co2p, logcval, fill_value = "extrapolate")
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

            int_fun[ialt] = interpolate.interp1d(co2p, logcval, fill_value = "extrapolate")
        else:
            raise ValueError('Not implemented for ndim = {}'.format(ndim))

    return int_fun, sign_coeff


def interp_coeff_linco2(coeffs, co2_profs):
    """
    Interpolates linearly the coeff against cco2.
    """

    ndim = coeffs[0].ndim
    int_fun = np.empty(coeffs[0].shape, dtype = object)

    n_alts = coeffs[0].shape[0]

    for ialt in range(n_alts):
        co2p = np.array([co[ialt] for co in co2_profs])
        if ndim == 2:
            for j in range(n_alts):
                cval = np.array([co[j, ialt] for co in coeffs])

                int_fun[j, ialt] = interpolate.interp1d(co2p, cval, fill_value = "extrapolate")
        elif ndim == 1:
            cval = np.array([co[ialt] for co in coeffs])

            int_fun[ialt] = interpolate.interp1d(co2p, cval, fill_value = "extrapolate")
        else:
            raise ValueError('Not implemented for ndim = {}'.format(ndim))

    return int_fun


def coeff_from_interp_log(int_fun, sign_coeff, co2_prof):
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


def coeff_from_interp_lin(int_fun, co2_prof):
    """
    Reconstructs the acoeff.
    """

    coeff = np.zeros(int_fun.shape)

    n_alts = int_fun.shape[0]
    ndim = int_fun.ndim

    for ialt in range(n_alts):
        if ndim == 1:
            interpcos = int_fun[ialt](co2_prof[ialt])
        else:
            interpcos = np.array([intfu(co2_prof[ialt]) for intfu in int_fun[..., ialt]])

        coeff[..., ialt] = interpcos

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


def delta_xi_at_x0_afit(xis, cco2, ialt, xis_b, atmweigths = atmweigths, all_coeffs = all_coeffs, hr_ref_nam = 'hr_ref', atm_pt = atm_pt, verbose = False):
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

    if verbose:
        print('------ xis:')
        print(xis)
        print('------ check hrs:')
        print(atm, hr, hr_somma)
        print('------ delta: ------')
        print(fu)
        print('-----------------------')

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


def jacdelta_xi_at_x0_afit(xis, cco2, ialt, xis_b, atmweigths = atmweigths, all_coeffs = all_coeffs, hr_ref_nam = 'hr_ref', atm_pt = atm_pt, verbose = False):
    """
    Jacobian of delta_xi_at_x0_afit.
    xis_b is not used, but the code expects the same parameters that are used by delta_xi_at_x0_afit
    """

    #print('atmweigths: ', atmweigths)

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
            ajac += (asurf[ialt] - agn_surf) * phi_fun_g
            # print(asurf[ialt], agn_surf, phi_fun_g)
            # print('aaaaa')
            # print(ajac, agn, acoeff[:, ialt], phi_fun)
            # print(np.sqrt(atmweigths[allatms[i]])/np.sum(xis))

            J[i,k] = np.sqrt(atmweigths[allatms[i]])/np.sum(xis) * ajac
            # sys.exit()

    if verbose:
        print('------ xis: ------')
        print(xis)
        print('------ Jacobian: ------')
        print(J)
        print('-----------------------')

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

#########################   REPARAM LOW  ##################################

def hr_reparam_low(cco2, temp, surf_temp, regrcoef = regrcoef, nlte_corr = nlte_corr, all_coeffs = all_coeffs, alt1_nlte = 40, alt2_nlte = 51):
    """
    Calculates cooling rate in LTE + low NLTE region.
    """

    surfanom = surf_temp - regrcoef['surfmean']
    atm_anom_mean = regrcoef['amean']
    eof0 = regrcoef['eof0']
    eof1 = regrcoef['eof1']
    n_alts = len(eof0)

    pc0 = np.dot(temp[:n_alts]-atm_anom_mean, eof0)
    pc1 = np.dot(temp[:n_alts]-atm_anom_mean, eof1)

    acoeff, bcoeff, asurf, bsurf = coeffs_from_eofreg(cco2, temp, surf_temp, method = '2eof', regrcoef = regrcoef)
    hr_new = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = 66)

    hra, hrb = hr_from_ab_diagnondiag(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts=66)
    hr_nlte_corr = nlte_corr[(cco2, 'c')] + nlte_corr[(cco2, 'm1')] * hra[alt1_nlte:alt2_nlte] + nlte_corr[(cco2, 'm2')] * hrb[alt1_nlte:alt2_nlte] + nlte_corr[(cco2, 'm3')] * pc0 + nlte_corr[(cco2, 'm4')] * pc1

    hr_new[alt1_nlte:alt2_nlte] = hr_new[alt1_nlte:alt2_nlte] + hr_nlte_corr
    hr_new[alt2_nlte:] = np.nan

    return hr_new


def hr_reparam_full(pres, temp, surf_temp, co2vmr, ovmr, o2vmr, n2vmr, regrcoef = regrcoef, nlte_corr = nlte_corr, alpha_fit = None, all_coeffs = all_coeffs, alt1_nlte = 40, alt2_nlte = 51):
    """
    Calculates cooling rate in LTE + low NLTE region.
    """

    surfanom = surf_temp - regrcoef['surfmean']
    atm_anom_mean = regrcoef['amean']
    eof0 = regrcoef['eof0']
    eof1 = regrcoef['eof1']
    n_alts = len(eof0)

    pc0 = np.dot(temp[:n_alts]-atm_anom_mean, eof0)
    pc1 = np.dot(temp[:n_alts]-atm_anom_mean, eof1)

    acoeff, bcoeff, asurf, bsurf = coeffs_from_eofreg(cco2, temp, surf_temp, method = '2eof', regrcoef = regrcoef)
    hr_new = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = 66)

    hra, hrb = hr_from_ab_diagnondiag(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts=66)
    hr_nlte_corr = nlte_corr[(cco2, 'c')] + nlte_corr[(cco2, 'm1')] * hra[alt1_nlte:alt2_nlte] + nlte_corr[(cco2, 'm2')] * hrb[alt1_nlte:alt2_nlte] + nlte_corr[(cco2, 'm3')] * pc0 + nlte_corr[(cco2, 'm4')] * pc1

    hr_new[alt1_nlte:alt2_nlte] = hr_new[alt1_nlte:alt2_nlte] + hr_nlte_corr
    hr_new[alt2_nlte:] = np.nan

    ### E ora la parte high
    lamb = calc_lamb(pres, temp, ovmr, o2vmr, n2vmr)
    MM = calc_MM(ovmr, o2vmr, n2vmr)

    hra = hra[alt2_nlte:n_top+1]
    hrb = hrb[alt2_nlte:n_top+1]
    # population upper state
    phifunz = np.exp(-E_fun/(kbc*temp[alt2_nlte:n_top+1]))
    lambdivA = lamb[alt2_nlte:n_top+1]/1.5988
    popup = lambdivA*phifunz

    atm_anom_mean = alpha_fit['amean']
    eof0 = alpha_fit['eof0']
    eof1 = alpha_fit['eof1']
    pc0 = np.dot(temp[alt2_nlte:n_top+1]-atm_anom_mean, eof0)
    pc1 = np.dot(temp[alt2_nlte:n_top+1]-atm_anom_mean, eof1)

    mod = 5
    alpha5 = alpha_fit[(cco2, 'c', mod)] + alpha_fit[(cco2, 'm1', mod)] * pc0 + alpha_fit[(cco2, 'm2', mod)] * pc1 + alpha_fit[(cco2, 'm3', mod)] * popup + alpha_fit[(cco2, 'm4', mod)] * hra + alpha_fit[(cco2, 'm5', mod)] * hrb

    #L_esc = AAAAAAAAAAAAAAAA

    hr_full = recformula(alpha5, L_esc, lamb, hr_new, co2vmr, MM, temp, n_alts_trlo = alt2_nlte, n_alts_trhi = n_top)

    # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    # manca L_esc, da mettere dipendente da cco2
    # e poi in realta cco2 qui non ci deve stare, deve dipendere solo da co2vmr

    return hr_new



###########################################################
# Upper trans region

def transrecformula(alpha, L_esc, lamb, eps125, co2vmr, MM, temp, n_trans = 7):
    """
    Recurrence formula in the upper transition region (with alpha).

    n_trans = n_alts_trhi-n_alts_trlo+1
    """
    eps125 = eps125 * cp_0 / (24*60*60)

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

    eps = eps * (24*60*60) / cp_0 # convert back to K/day

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


def delta_alpha_rec2(alpha, cco2, cose_upper_atm, n_alts_trlo = 50, n_alts_trhi = 56, weigths = np.ones(len(allatms)), all_coeffs = None, atm_pt = atm_pt, name_escape_fun = 'L_esc'):
    """
    This is done for all n_trans = 6 altitudes at a time.
    """

    n_trans = n_alts_trhi-n_alts_trlo+1

    fu = []
    for i, atm in enumerate(allatms):
        hr_ref = all_coeffs[(atm, cco2, 'hr_nlte')][n_alts_trlo:n_alts_trhi]

        L_esc = cose_upper_atm[(atm, cco2, name_escape_fun)][n_alts_trlo-1:n_alts_trhi]
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


def delta_alpha_rec2_atm(alpha, atm, cco2, cose_upper_atm, n_alts_trlo = 50, n_alts_trhi = 56, weigths = np.ones(len(allatms)), all_coeffs = None, atm_pt = atm_pt, name_escape_fun = 'L_esc'):
    """
    As rec2, but for a single atmosphere.
    """

    n_trans = n_alts_trhi-n_alts_trlo+1

    hr_ref = all_coeffs[(atm, cco2, 'hr_nlte')][n_alts_trlo:n_alts_trhi]

    L_esc = cose_upper_atm[(atm, cco2, name_escape_fun)][n_alts_trlo-1:n_alts_trhi]
    lamb = cose_upper_atm[(atm, cco2, 'lamb')][n_alts_trlo-1:n_alts_trhi]
    co2vmr = cose_upper_atm[(atm, cco2, 'co2vmr')][n_alts_trlo-1:n_alts_trhi]
    MM = cose_upper_atm[(atm, cco2, 'MM')][n_alts_trlo-1:n_alts_trhi]
    temp = atm_pt[(atm, 'temp')][n_alts_trlo-1:n_alts_trhi]
    eps125 = cose_upper_atm[(atm, cco2, 'eps125')]

    hr_calc = transrecformula(alpha, L_esc, lamb, eps125, co2vmr, MM, temp, n_trans = n_alts_trhi-n_alts_trlo+1)

    # atmweights will be squared by the loss function inside least_quares
    fu = hr_calc - hr_ref

    return fu


def delta_alpha_rec3(alpha, cco2, cose_upper_atm, n_alts_trlo = 50, n_alts_trhi = 56, weigths = np.ones(len(allatms)), all_coeffs = None, atm_pt = atm_pt, name_escape_fun = 'L_esc'):
    """
    This is done for all n_trans = 6 altitudes at a time.
    """

    n_trans = n_alts_trhi-n_alts_trlo+1

    fu = []
    for i, atm in enumerate(allatms):
        hr_ref = all_coeffs[(atm, cco2, 'hr_nlte')][n_alts_trlo:n_alts_trhi]

        L_esc = cose_upper_atm[(atm, cco2, name_escape_fun)][n_alts_trlo-1:n_alts_trhi]
        lamb = cose_upper_atm[(atm, cco2, 'lamb')][n_alts_trlo-1:n_alts_trhi]
        co2vmr = cose_upper_atm[(atm, cco2, 'co2vmr')][n_alts_trlo-1:n_alts_trhi]
        MM = cose_upper_atm[(atm, cco2, 'MM')][n_alts_trlo-1:n_alts_trhi]
        temp = atm_pt[(atm, 'temp')][n_alts_trlo-1:n_alts_trhi]
        eps125 = cose_upper_atm[(atm, cco2, 'eps125')]

        hr_calc = transrecformula(alpha, L_esc, lamb, eps125, co2vmr, MM, temp, n_trans = n_alts_trhi-n_alts_trlo+1)

        # atmweights will be squared by the loss function inside least_quares
        fu.append(hr_calc - hr_ref)

    #fu = np.concatenate(fu)
    fu = np.stack(fu)
    fu = weigths[:, np.newaxis]*fu**2
    fu = np.sqrt(np.sum(fu)) # in questo modo fu ha dimensione n_trans
    # #resid = np.sqrt(atmweigths[i] * np.sum((hr_calc - hr)**2))

    return fu


def calc_lamb(pres, temp, ovmr, o2vmr, n2vmr):
    """
    Calculates the lambda used in the transition formula.
    """
    n_dens = sbm.num_density(pres, temp)

    ###################### Rate coefficients ######################
    t13 = temp**(-1./3)

    # Collisional rate between CO2 and O:
    zo = 3.5e-13*np.sqrt(temp)+2.32e-9*np.exp(-76.75*t13) # use Granada parametrization
    #ZCO2O = KO Fomichev value
    # Collisional rates between CO2 and N2/O2:
    zn2=7e-17*np.sqrt(temp)+6.7e-10*np.exp(-83.8*t13)
    zo2=7e-17*np.sqrt(temp)+1.0e-9*np.exp(-83.8*t13)

    lamb = 1.5988/(1.5988 + n_dens*(n2vmr*zn2 + o2vmr*zo2 + ovmr*zo))

    return lamb


def calc_co2column(alts, pres, temp, co2vmr):
    """
    Calculates CO2 column above a certain point. (to be used for L_escape)
    """

    n_dens = num_density(pres, temp)
    n_co2 = n_dens * co2vmr

    #uok = []
    uok2 = []

    nco2spl = interp1d(alts, np.log(n_co2), fill_value = 'extrapolate')
    morealts = np.linspace(alts[0], 200., 1000)
    morenco2 = np.exp(nco2spl(morealts))
    for ial in range(len(alts)):
        #uok.append(np.trapz(n_co2[ial:], 1.e5*alts[ial:])) # integro in cm, voglio la colonna in cm-2
        alok = morealts >= alts[ial]
        uok2.append(np.trapz(morenco2[alok], 1.e5*morealts[alok])) # integro in cm, voglio la colonna in cm-2

    #utop = uok[-2] # Setting upper column equal to last step
    print('utop = {:7.2e}'.format(uok2[-1]))

    #uok = np.array(uok)
    uok2 = np.array(uok2)

    return uok2


def calc_co2column_P(pres, co2vmr, MM, extrapolate = True, minlogP = -14):
    """
    Calculates CO2 column above a certain point in pressure. Assumes hydrostatically stable column.
    """

    Nav = 6.02e23
    g_grav = 9.81

    kost = -100*1000*Nav/g_grav # pres to Pa, MM to kg

    if extrapolate:
        prlog = np.log(pres)
        p2 = np.append(prlog, np.arange(prlog[-1], minlogP, prlog[-1]-prlog[-2]))
        p2ex = np.exp(p2)

        nco2spl = interp1d(prlog, co2vmr, fill_value = 'extrapolate')
        morenco2 = nco2spl(p2)
        morenco2[morenco2 < 0] = 0.

        nmmspl = interp1d(prlog, MM, fill_value = 'extrapolate')
        moremm = nmmspl(p2)

        uok = []
        for ial in range(len(pres)):
            #alok = p2ex < pres[ial]
            uok.append(kost*np.trapz(morenco2[ial:]/moremm[ial:], p2ex[ial:])) # faccio tutto in SI
    else:
        uok = []
        for ial in range(len(pres)):
            uok.append(kost*np.trapz(co2vmr[ial:]/MM[ial:], pres[ial:])) # faccio tutto in SI

    uok = np.array(uok) * 1e-4 # to cm-2
    if extrapolate:
        print('utop = {:7.2e}'.format(uok[-1]))

    return uok


def num_density(P, T, vmr = 1.0):
    """
    Calculates num density. P in hPa, T in K, vmr in absolute fraction (not ppm!!)
    """
    n = P*vmr/(T*kb) # num. density in cm-3

    return n

def calc_MM(ovmr, o2vmr, n2vmr):
    MM = (n2vmr*28+o2vmr*32+ovmr*16)/(n2vmr+o2vmr+ovmr) # Molecular mass
    return MM

def calc_cp(MM, ovmr):
    cp = 8.31441e7/MM*(7./2.*(1.-ovmr)+5./2.*ovmr)
    return cp

def recformula(alpha, L_esc, lamb, hr, co2vmr, MM, temp, n_alts_trlo = 50, n_alts_trhi = 56, n_alts_cs = 65, ovmr = None, debug = False):
    """
    Recurrence formula in the upper transition region (with alpha).

    With full vectors.
    """
    # n_alts = len(hr)
    # hr_new = hr.copy()
    n_alts = len(temp)
    hr_new = np.zeros(len(temp))
    hr_new[:len(hr)] = hr[:]

    phi_fun = np.exp(-E_fun/(kbc*temp))

    # MMfom = np.ones(len(hr))
    # MMfom[:50] = 28.96
    # pruz = np.array([28.95,28.94,28.93,28.90,28.87,28.82,28.76,28.69,28.61,28.52,28.40,28.25,28.08,27.89,27.69,27.48,27.27,27.06,26.83,26.55,26.20])
    # spl = spline(np.arange(len(pruz)), pruz)
    # pruz_ok = spl(np.arange(len(hr)-50))
    # MMfom[50:] = pruz_ok
    # MM = MMfom

    if ovmr is not None:
        cp = calc_cp(MM, ovmr)
    else:
        cp = np.ones(n_alts)*cp_0

    if debug: print('cp', cp)

    eps125 = hr[n_alts_trlo-1] * cp[n_alts_trlo-1] / (24*60*60)

    alpha_ok = np.ones(n_alts)
    alpha_ok[n_alts_trlo-1:n_alts_trhi] = alpha
    dj = L_esc*alpha_ok
    if debug: print('alpha', alpha_ok[n_alts_trlo-1:])
    if debug: print('L_corr', dj[n_alts_trlo-1:])

    #fac = (2.63187e11 * co2vmr * (1-lamb))/MM
    fac = (2.55520997e11 *co2vmr * (1-lamb))/MM

    eps_gn = np.zeros(n_alts)
    #eps_gn[n_alts_trlo-1] = 1.10036e-10*eps125/(co2vmr[n_alts_trlo-1] * (1-lamb[n_alts_trlo-1])) ### should change sign to be consistent with fomichev's (cooling rate)?
    eps_gn[n_alts_trlo-1] = eps125/fac[n_alts_trlo-1]

    for j in range(n_alts_trlo, n_alts_cs): # Formula 9
        Djj = 0.25*(dj[j-1] + 3*dj[j])
        Djjm1 = 0.25*(dj[j] + 3*dj[j-1])

        Fj = (1 - lamb[j]*(1-Djj))
        Fjm1 = (1 - lamb[j-1]*(1-Djjm1))

        #print(j, Djj, Djjm1, Fj, Fjm1)
        eps_gn[j] = (Fjm1*eps_gn[j-1] + Djjm1*phi_fun[j-1] - Djj*phi_fun[j])/Fj

    #c --- the reccurence formula
          # do 11 I=2,17
          #     im=i-1
          #     aa1=1.-lambda(im)*(1.-.25*AL(i)-.75*AL(im))
          #     aa2=1.-lambda(i)*(1.-.75*AL(i)-.25*AL(im))
          #     d1=-.25*(AL(i)+3.*AL(im))
          #     d2=.25*(3.*AL(i)+AL(im))
          #     h2=(aa1*h1-d1*su(im+50)-d2*su(i+50))/aa2
          #     H(i+42)=h2*CO2(i)*(1.-lambda(i))/AM(i)*const
          #     h1=h2
          # end do


    h1 = eps_gn[n_alts_trlo-1]
    for j in range(n_alts_trlo, n_alts_cs):
        aa1=1.-lamb[j-1]*(1.-.25*dj[j]-.75*dj[j-1])
        aa2=1.-lamb[j]*(1.-.75*dj[j]-.25*dj[j-1])
        d1=-.25*(dj[j]+3.*dj[j-1])
        d2=.25*(3.*dj[j]+dj[j-1])
        h2=(aa1*h1-d1*phi_fun[j-1]-d2*phi_fun[j])/aa2
        eps_gn[j] = h2
        if debug:
            #print(j-n_alts_trlo, aa1, aa2, d1, d2, h1, h2, co2vmr[j], lamb[j], MM[j], fac[j]*h2)
            print(fac[j]*h2)
        h1 = h2

    if debug:
        print('\n')
        for j in range(n_alts_trlo, n_alts_cs):
            print(j, eps_gn[j], fac[j]*eps_gn[j])
    ##### HERE the cool-to-space part
    # now for the cs region:
    if n_alts > n_alts_cs:
        Phi_165 = eps_gn[n_alts_cs] + phi_fun[n_alts_cs]
        eps_gn[n_alts_cs:] = (Phi_165 - phi_fun[n_alts_cs:])
        #eps[n_alts_cs:] = fac[n_alts_cs:] * (Phi_165 - phi_fun[j])

    if debug:
        print(n_alts_cs)
        print('\n')
        for j in range(n_alts_trlo, n_alts):
            print(j, eps_gn[j], fac[j]*eps_gn[j])

    hr_new[n_alts_trlo:] = fac[n_alts_trlo:] * eps_gn[n_alts_trlo:]  # Formula 7 ### change sign back to heating rate if changed above

    if debug:
        print('\n')
        for j in range(n_alts_trlo, n_alts):
            print(j, eps_gn[j], fac[j]*eps_gn[j], hr_new[j])
    # for j in range(n_alts_trlo, n_alts):
    #     print(j, hr_new[j])

    hr_new[n_alts_trlo:] = hr_new[n_alts_trlo:] * (24*60*60) / cp[n_alts_trlo:] # convert back to K/day
    # for j in range(n_alts_trlo, n_alts):
    #     print(j, hr_new[j])

    return hr_new


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


def plot_coeff(coeff, n_alts = 51, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize = (16,12))

    for ialt, col in zip(range(n_alts), color_set(n_alts)):
        ax.plot(coeff[:n_alts, ialt], np.arange(n_alts), color = col)

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
