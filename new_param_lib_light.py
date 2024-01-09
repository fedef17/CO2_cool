#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

#from matplotlib import pyplot as plt
#import matplotlib.cm as cm

from scipy import io
import scipy.constants as const
from scipy import interpolate
from scipy.interpolate import PchipInterpolator as spline
from scipy.interpolate import interp1d

#import pickle

############################################################

# Constants
kbc = const.k/(const.h*100*const.c) # 0.69503
kboltz = 1.38064853e-23 # J/K
kb = 1.38065e-19 # boltzmann constant to use with P in hPa, n in cm-3, T in K (10^-4 * Kb standard)
E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition
cp_0 = 1.005e7 # specific enthalpy dry air - erg g-1 K-1


# Define global parameters
n_alts_all = 83 # vertical size of reference atmosphere
n_co2prof = 8 # number of reference co2 profiles
max_alts_curtis = 55 # max altitude for calculation with curtis matrix. 

vfit = 'vf5'
afit = 'a0s'
n_top = 65

thisdir = os.path.dirname(os.path.abspath(__file__)) + '/'

ctag = '{}-{}-{}'.format(vfit, afit, n_top)
#coeff_dir = thisdir + 'data/'
#coeff_file = thisdir + 'data/coeffs_finale_{}.p'.format(ctag)

#############################################################

def new_param_full_v1(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, coeffs = None, ctag = ctag, interp_coeffs = None, max_alts = max_alts_curtis, extrap_co2col = True, debug_alpha = None, alt2 = 51, n_top = 65, n_alts_cs = 80, debug = False, zofac = 1.):
    """
    New param valid for the full atmosphere.
    """
    # alt2 = 51 # 50
    # n_top = 65 # 61
    # n_alts_cs = 80

    ##### Interpolate all profiles to param grid.
    #print('I am not interpolating yet! profiles should already be given on a fix grid')


    ### Interpolation of the coefficients to the actual CO2 profile
    if interp_coeffs is None:
        if coeffs is None:
            #coeffs = pickle.load(open(coeff_file, 'rb'))
            coeffs = dict()
            for ke in ['uco2', 'bsurf', 'co2profs', 'asurf', 'Lesc', 'alpha', 'acoeff', 'bcoeff']:
                coeffs[ke] = np.load(thisdir + 'data/coeffs_{}_{}.npy'.format(ctag, ke))
            
        print('interpolating for co2! this should be done calling npl.precalc_interp_v1() just once')
        interp_coeffs = precalc_interp_v1(coeffs = coeffs, ctag = ctag)

    debudict = dict()

    print('Coeffs from interpolation!')
    calc_coeffs = dict()
    for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        int_fun = interp_coeffs[(nam, 'int_fun')]
        sc = interp_coeffs[(nam, 'signc')]

        coeff = coeff_from_interp_log(int_fun, sc, co2vmr)
        calc_coeffs[nam] = coeff

    calc_coeffs['alpha'] = coeff_from_interp_lin(interp_coeffs[('alpha', 'int_fun')], co2vmr[alt2:n_top+1])

    calc_coeffs['L_all'] = coeff_from_interp_lin(interp_coeffs[('Lesc', 'int_fun')], co2vmr, use_co2mean = True)
    calc_coeffs['uco2'] = interp_coeffs['uco2']

    #####################################################

    hr_calc = hr_from_ab(calc_coeffs['acoeff'], calc_coeffs['bcoeff'], calc_coeffs['asurf'], calc_coeffs['bsurf'], temp, surf_temp, max_alts = max_alts)

    Lspl_all = spline(calc_coeffs['uco2'], calc_coeffs['L_all'], extrapolate = False)
    MM = calc_MM(ovmr, o2vmr, n2vmr)
    uok2 = calc_co2column_P(pres, co2vmr, MM, extrapolate = extrap_co2col)

    L_esc = Lspl_all(uok2)
    L_esc[:30][np.isnan(L_esc[:30])] = 0.0 # for extrapolated regions
    L_esc[-20:][np.isnan(L_esc[-20:])] = 1.0 # for extrapolated regions
    if np.any(np.isnan(L_esc)):
        print(uok2)
        print(L_esc)
        raise ValueError('{} nans in L_esc!'.format(np.sum(np.isnan(L_esc))))

    lamb = calc_lamb(pres, temp, ovmr, o2vmr, n2vmr, zofac = zofac)

    debudict['L_esc'] = L_esc
    debudict['MM'] = MM
    debudict['lamb'] = lamb

    if debug_alpha is not None:
        print('alpha old: ', calc_coeffs['alpha'])
        alpha = debug_alpha
        print('alpha new: ', alpha)
    else:
        alpha = calc_coeffs['alpha']

    debudict['alpha'] = alpha

    hr_calc = recformula(alpha, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, n_alts_cs = n_alts_cs, ovmr = ovmr, factor_from_code = True)

    if debug:
        return hr_calc, debudict
    else:
        return hr_calc


def precalc_interp_v1(coeffs = None, ctag = ctag, alt2 = 51, n_top = 65):

    if coeffs is None:
        #coeffs = pickle.load(open(coeff_file, 'rb'))
        coeffs = dict()
        for ke in ['uco2', 'bsurf', 'co2profs', 'asurf', 'Lesc', 'alpha', 'acoeff', 'bcoeff']:
            coeffs[ke] = np.load(thisdir + 'data/coeffs_{}_{}.npy'.format(ctag, ke))

    interp_coeffs = dict()
    for nam in ['acoeff', 'bcoeff', 'asurf', 'bsurf']:
        print(nam)
        int_fun, signc = interp_coeff_logco2(coeffs[nam], coeffs['co2profs'])
        interp_coeffs[(nam, 'int_fun')] = int_fun
        interp_coeffs[(nam, 'signc')] = signc

    Lesc_all = coeffs['Lesc']
    int_fun = interp_coeff_linco2(Lesc_all, coeffs['co2profs'], use_co2mean = True)
    interp_coeffs[('Lesc', 'int_fun')] = int_fun
    interp_coeffs['uco2'] = coeffs['uco2']

    int_fun = interp_coeff_linco2(coeffs['alpha'], coeffs['co2profs'][:, alt2:n_top+1])
    interp_coeffs[('alpha', 'int_fun')] = int_fun

    return interp_coeffs


def new_param_full_allgrids_v1(temp, surf_temp, pres, co2vmr, ovmr, o2vmr, n2vmr, coeffs = None, ctag = ctag, interp_coeffs = None, debug_Lesc = None, debug_alpha = None, debug = False, debug_co2interp = None, debug_allgr = False, extrap_co2col = True, debug_starthigh = None, alt2up = 51, n_top = 65, zofac = 1.):
    """
    Wrapper for new_param_full that takes in input vectors on arbitrary grids.
    """
    
    print('USING NEW PARAM with Fomichev approach')

    if interp_coeffs is None:
        print('Precalculate interp function for faster calculations')
        interp_coeffs = precalc_interp_v1(coeffs = coeffs, ctag = ctag, n_top = n_top)

    ## custom x grid
    x = np.log(1000./pres)

    ## reference x grid
    x_ref_max = 20.625
    if np.max(x) > x_ref_max:
        x_ref = np.arange(0.125, np.max(x) + 0.001, 0.25)
    else:
        x_ref = np.arange(0.125, x_ref_max + 0.001, 0.25)

    ##### INTERPOLATE EVERYTHING TO REFERENCE GRID HERE ####
    spl = spline(x, temp, extrapolate = True)
    temp_rg = spl(x_ref)
    temp_rg[x_ref > np.max(x)+0.25] = np.nan

    spl = spline(x, np.log(pres), extrapolate = True)
    pres_rg = spl(x_ref)
    pres_rg = np.exp(pres_rg)
    pres_rg[x_ref > np.max(x)+0.25] = np.nan

    spl = spline(x, ovmr, extrapolate = True)
    ovmr_rg = spl(x_ref)
    ovmr_rg[x_ref > np.max(x)+0.25] = np.nan

    spl = spline(x, o2vmr, extrapolate = True)
    o2vmr_rg = spl(x_ref)
    o2vmr_rg[x_ref > np.max(x)+0.25] = np.nan

    spl = spline(x, co2vmr, extrapolate = True)
    co2vmr_rg = spl(x_ref)
    co2vmr_rg[x_ref > np.max(x)+0.25] = np.nan

    spl = spline(x, n2vmr, extrapolate = True)
    n2vmr_rg = spl(x_ref)
    n2vmr_rg[x_ref > np.max(x)+0.25] = np.nan


    ########## Call new param
    resu = new_param_full_v1(temp_rg, surf_temp, pres_rg, co2vmr_rg, ovmr_rg, o2vmr_rg, n2vmr_rg, coeffs = coeffs, ctag = ctag, interp_coeffs = interp_coeffs, extrap_co2col = extrap_co2col, debug_alpha = debug_alpha, alt2 = alt2up, n_top = n_top, debug = debug, zofac = zofac)

    if debug:
        hr_calc_fin, cose = resu
    else:
        hr_calc_fin = resu

    ##### INTERPOLATE OUTPUT TO ORIGINAL GRID ####

    spl = spline(x_ref, hr_calc_fin, extrapolate = True)
    hr_calc = spl(x)

    if debug:
        return hr_calc, cose
    elif debug_allgr:
        return hr_calc, [x, x_ref, temp_rg, pres_rg, ovmr_rg, co2vmr_rg]
    else:
        return hr_calc


def hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, max_alts = max_alts_curtis):# max_alts was 51, but n_alts_all in new_param_full
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


def interp_coeff_linco2(coeffs, co2_profs, use_co2mean = False):
    """
    Interpolates linearly the coeff against cco2.
    """

    ndim = coeffs[0].ndim
    int_fun = np.empty(coeffs[0].shape, dtype = object)

    n_alts = coeffs[0].shape[0]

    for ialt in range(n_alts):
        if use_co2mean:
            co2p = np.array([np.mean(co[:40]) for co in co2_profs])
        else:
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


def coeff_from_interp_lin(int_fun, co2_prof, use_co2mean = False):
    """
    Reconstructs the acoeff.
    """

    coeff = np.zeros(int_fun.shape)

    n_alts = int_fun.shape[0]
    ndim = int_fun.ndim

    for ialt in range(n_alts):
        if use_co2mean:
            co = np.nanmean(co2_prof[:40])
        else:
            co = co2_prof[ialt]

        if ndim == 1:
            interpcos = int_fun[ialt](co)
        else:
            interpcos = np.array([intfu(co) for intfu in int_fun[..., ialt]])

        coeff[..., ialt] = interpcos

    return coeff


def calc_lamb(pres, temp, ovmr, o2vmr, n2vmr, zofac = 1.):
    """
    Calculates the lambda used in the transition formula.
    """
    n_dens = num_density(pres, temp)

    ###################### Rate coefficients ######################
    t13 = temp**(-1./3)

    # Collisional rate between CO2 and O:
    zo = zofac*3.5e-13*np.sqrt(temp)+2.32e-9*np.exp(-76.75*t13) # use Granada parametrization
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
    #print('utop = {:7.2e}'.format(uok2[-1]))

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

    fnan = None
    if np.any(np.isnan(pres)):
        fnan = np.where(np.isnan(pres))[0][0] # end of atmosphere

    if extrapolate:
        prlog = np.log(pres[:fnan])
        p2 = np.append(prlog, np.arange(prlog[-1], minlogP, prlog[-1]-prlog[-2])[1:]) ### the increments in the reference log pressure are constant, this is equal to pres_rg up to end of reference grid
        p2ex = np.exp(p2)

        # Linear extrapolation of co2vmr in log pressure
        nco2spl = interp1d(prlog, co2vmr[:fnan], fill_value = 'extrapolate')
        morenco2 = nco2spl(p2)
        morenco2[morenco2 < 0] = 0.

        # Linear extrapolation of MM in log pressure
        nmmspl = interp1d(prlog, MM[:fnan], fill_value = 'extrapolate')
        moremm = nmmspl(p2)
        moremm[moremm < 20] = 20.


        uok = []
        for ial in range(len(pres)):
            #alok = p2ex < pres[ial]
            uok.append(kost*np.trapz(morenco2[ial:]/moremm[ial:], p2ex[ial:])) # faccio tutto in SI
    else:
        uok = []
        for ial in range(len(pres)):
            uok.append(kost*np.trapz(co2vmr[ial:fnan]/MM[ial:fnan], pres[ial:fnan])) # faccio tutto in SI

    uok = np.array(uok) * 1e-4 # to cm-2
    #if extrapolate:
    #    print('utop = {:7.2e}'.format(uok[-1]))

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

def recformula(alpha, L_esc, lamb, hr, co2vmr, MM, temp, n_alts_trlo = 51, n_alts_trhi = 65, n_alts_cs = 80, ovmr = None, debug = False, factor_from_code = True, debug_starthigh = None):
    """
    n_alts_trlo, n_alts_trhi, n_alts_cs were: 50, 56, 65
    Recurrence formula in the upper transition region (with alpha).

    With full vectors.
    """

    n_alts = len(temp)
    hr_new = np.zeros(len(temp))
    hr_new[:len(hr)] = hr[:]

    phi_fun = np.exp(-E_fun/(kbc*temp))

    if ovmr is not None:
        cp = calc_cp(MM, ovmr)
    else:
        print('WARNING!! using constant cp in upper atmosphere!')
        cp = np.ones(n_alts)*cp_0

    if debug: print('cp', cp)

    if debug_starthigh is not None:
        print('debug_starthigh, setting eps125 to ref value: {}'.format(debug_starthigh))
        eps125 = debug_starthigh * cp[n_alts_trlo-1] / (24*60*60)
    else:
        eps125 = hr[n_alts_trlo-1] * cp[n_alts_trlo-1] / (24*60*60)

    alpha_ok = np.ones(n_alts)
    alpha_ok[n_alts_trlo-1:n_alts_trhi] = alpha
    dj = L_esc*alpha_ok
    if debug: print('alpha', alpha_ok[n_alts_trlo-1:])
    if debug: print('L_corr', dj[n_alts_trlo-1:])

    if factor_from_code:
        numfac = 2.55520997e11
    else:
        numfac = 2.63187e11

    #fac = (2.63187e11 * co2vmr * (1-lamb))/MM
    fac = (numfac * co2vmr * (1-lamb))/MM

    eps_gn = np.zeros(n_alts)
    #eps_gn[n_alts_trlo-1] = 1.10036e-10*eps125/(co2vmr[n_alts_trlo-1] * (1-lamb[n_alts_trlo-1]))
    eps_gn[n_alts_trlo-1] = eps125/fac[n_alts_trlo-1]

    for j in range(n_alts_trlo, n_alts): # Formula 9
        Djj = 0.25*(dj[j-1] + 3*dj[j])
        Djjm1 = 0.25*(dj[j] + 3*dj[j-1])

        Fj = (1 - lamb[j]*(1-Djj))
        Fjm1 = (1 - lamb[j-1]*(1-Djjm1))

        #print(j, Djj, Djjm1, Fj, Fjm1)
        eps_gn[j] = (Fjm1*eps_gn[j-1] + Djjm1*phi_fun[j-1] - Djj*phi_fun[j])/Fj

    if debug:
        print('\n')
        for j in range(n_alts_trlo, n_alts_cs):
            print(j, eps_gn[j], fac[j]*eps_gn[j])
    
    ##### HERE the cool-to-space part
    # now for the cs region:
    if n_alts > n_alts_cs:
        Phi_165 = eps_gn[n_alts_cs] + phi_fun[n_alts_cs]
        eps_gn[n_alts_cs:] = (Phi_165 - phi_fun[n_alts_cs:])

    if debug:
        print(n_alts_cs)
        print('\n')
        for j in range(n_alts_trlo, n_alts):
            print(j, eps_gn[j], fac[j]*eps_gn[j])

    hr_new[n_alts_trlo:] = fac[n_alts_trlo:] * eps_gn[n_alts_trlo:]  # Formula 7

    if debug:
        print('\n')
        for j in range(n_alts_trlo, n_alts):
            print(j, eps_gn[j], fac[j]*eps_gn[j], hr_new[j])

    hr_new[n_alts_trlo:] = hr_new[n_alts_trlo:] * (24*60*60) / cp[n_alts_trlo:] # convert back to K/day

    return hr_new


###########################################################
## Plotting functions (not needed now)

# def plot_pdfpages(filename, figs, save_single_figs = False, fig_names = None):
#     """
#     Saves a list of figures to a pdf file.
#     """
#     from matplotlib.backends.backend_pdf import PdfPages

#     pdf = PdfPages(filename)
#     for fig in figs:
#         pdf.savefig(fig)
#     pdf.close()

#     if save_single_figs:
#         indp = filename.index('.')
#         cartnam = filename[:indp]+'_figures/'
#         if not os.path.exists(cartnam):
#             os.mkdir(cartnam)
#         if fig_names is None:
#             fig_names = ['pag_{}'.format(i+1) for i in range(len(figs))]
#         for fig,nam in zip(figs, fig_names):
#             fig.savefig(cartnam+nam+'.pdf')

#     return


# def plot_coeff(coeff, n_alts = 51, ax = None):
#     if ax is None:
#         fig, ax = plt.subplots(figsize = (16,12))

#     for ialt, col in zip(range(n_alts), color_set(n_alts)):
#         ax.plot(coeff[:n_alts, ialt], np.arange(n_alts), color = col)

#     return


# def manuel_plot(y, xs, labels, xlabel = None, ylabel = None, title = None, xlimdiff = None, colors = None, linestyles = None, xlim = (None, None), ylim = (None, None), orizlines = [70., 85.], linewidth = 1.):
#     """
#     Plots plt.plot(x, y, lab) for each x in xs. Plots the differences of all xs wrt xs[0] in a side plot.
#     """
#     fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
#     if colors is None: colors = color_set(len(xs))
#     if linestyles is None: linestyles = len(xs)*['-']
#     i = 0
#     for x, lab, col, lst in zip(xs, labels, colors, linestyles):
#         a0.plot(x, y, label = lab, color = col, linestyle = lst, linewidth = linewidth)
#         if i == 0:
#             i+=1
#             continue
#         if i == 1:
#             a1.axvline(0., color = 'grey', alpha = 0.6)
#             a1.axvline(0.5, color = 'grey', alpha = 0.4, linestyle = ':', linewidth = 0.8)
#             a1.axvline(-0.5, color = 'grey', alpha = 0.4, linestyle = ':', linewidth = 0.8)
#             a1.axvline(1.0, color = 'grey', alpha = 0.4, linestyle = '--', linewidth = 0.8)
#             a1.axvline(-1.0, color = 'grey', alpha = 0.4, linestyle = '--', linewidth = 0.8)
#         a1.plot(x - xs[0], y, color = col, linestyle = lst, linewidth = linewidth)
#         i+=1

#     for orizli, col in zip(orizlines, ['red', 'orange', 'green', 'blue']):
#         a0.axhline(orizli, color = col, alpha = 0.6, linestyle = '--')
#         a1.axhline(orizli, color = col, alpha = 0.6, linestyle = '--')
#     a0.grid()
#     a1.grid()
#     if xlimdiff is not None:
#         a1.set_xlim(xlimdiff)
#     a0.legend(loc = 3)

#     if xlim is not None:
#         a0.set_xlim(xlim)

#     if ylim is not None:
#         a0.set_ylim(ylim)
#         a1.set_ylim(ylim)

#     if xlabel is not None: a0.set_xlabel(xlabel)
#     if ylabel is not None: a0.set_ylabel(ylabel)
#     if title is not None: a0.set_title(title)

#     #fig.tight_layout()

#     return fig, a0, a1


# def adjust_ax_scale(axes, sel_axis = 'both'):
#     """
#     Given a set of axes, uniformizes the scales.
#     < sel_axis > : 'x', 'y' or 'both' (default)
#     """

#     if sel_axis == 'x' or sel_axis == 'both':
#         limits_min = []
#         limits_max = []
#         for ax in axes:
#             limits_min.append(ax.get_xlim()[0])
#             limits_max.append(ax.get_xlim()[1])

#         maxlim = np.max(limits_max)
#         minlim = np.min(limits_min)
#         for ax in axes:
#             ax.set_xlim((minlim,maxlim))

#     if sel_axis == 'y' or sel_axis == 'both':
#         limits_min = []
#         limits_max = []
#         for ax in axes:
#             limits_min.append(ax.get_ylim()[0])
#             limits_max.append(ax.get_ylim()[1])

#         maxlim = np.max(limits_max)
#         minlim = np.min(limits_min)
#         for ax in axes:
#             ax.set_ylim((minlim,maxlim))

#     return


# def color_set(n, cmap = 'nipy_spectral', full_cb_range = False):
#     """
#     Gives a set of n well chosen (hopefully) colors, darker than bright_thres. bright_thres ranges from 0 (darker) to 1 (brighter).

#     < full_cb_range > : if True, takes all cb values. If false takes the portion 0.05/0.95.
#     """
#     cmappa = cm.get_cmap(cmap)
#     colors = []

#     if full_cb_range:
#         valori = np.linspace(0.0,1.0,n)
#     else:
#         valori = np.linspace(0.05,0.95,n)

#     for cos in valori:
#         colors.append(cmappa(cos))

#     return colors
