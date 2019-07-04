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
import pickle

sys.path.insert(0, '/home/fedefab/Scrivania/Research/Dotto/Git/spect_robot/')
sys.path.insert(0, '/home/fedefab/Scrivania/Research/Dotto/Git/pythall/')
import spect_base_module as sbm
import spect_classes as spcl

kbc = const.k/(const.h*100*const.c) # 0.69503
kboltz = 1.38064853e-23 # J/K
E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition

cp = 1.005e7 # specific enthalpy dry air - erg g-1 K-1
#############################################################

cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'

allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
atmweigths = [0.3, 0.1, 0.1, 0.4, 0.05, 0.05]
atmweigths = dict(zip(allatms, atmweigths))
allco2 = np.arange(1,7)

all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v1.p'))
atm_pt = pickle.load(open(cart_out + 'atm_pt.p'))
n_alts = 40

from scipy.optimize import Bounds, minimize, least_squares

#############################################################

def hr_atm_calc(atm, cco2):
    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    acoeff = all_coeffs[(atm, cco2, 'acoeff')]
    bcoeff = all_coeffs[(atm, cco2, 'bcoeff')]
    asurf = all_coeffs[(atm, cco2, 'asurf')]
    bsurf = all_coeffs[(atm, cco2, 'bsurf')]

    hr = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)

    return hr

def hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp):
    n_alts = len(temp)
    epsilon_ab_tot = np.zeros(n_alts, dtype = float)

    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    for xi in range(n_alts):
        epsilon_ab_tot[xi] = np.sum((acoeff[:, xi] + bcoeff[:, xi]* phi_fun[xi]) * phi_fun) # il contributo della colonna
        epsilon_ab_tot[xi] += (asurf[xi] + bsurf[xi]* phi_fun[xi]) * phi_fun_g

    return epsilon_ab_tot


def hr_from_ab_at_x0(acoeff, bcoeff, asurf, bsurf, temp, surf_temp, x0):
    phi_fun = np.exp(-E_fun/(kbc*temp))
    phi_fun_g = np.exp(-E_fun/(kbc*surf_temp))

    # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
    epsilon_ab_tot = np.sum((acoeff[:, x0] + bcoeff[:, x0]* phi_fun[x0]) * phi_fun) # il contributo della colonna
    epsilon_ab_tot += (asurf[x0] + bsurf[x0]* phi_fun[x0]) * phi_fun_g

    return epsilon_ab_tot


def hr_from_xi(xis, atm, cco2, all_coeffs = all_coeffs, atm_pt = atm_pt, allatms = allatms, n_alts = 40):
    """
    Calculates the HR from the acoeff and bcoeff of the different atmospheres, using the weights xis.
    """

    temp = atm_pt[(atm, 'temp')]
    surf_temp = atm_pt[(atm, 'surf_temp')]

    hr_somma = np.zeros(n_alts, dtype = float)
    for atmprim, xi in zip(allatms, xis):
        acoeff = all_coeffs[(atmprim, cco2, 'acoeff')]
        bcoeff = all_coeffs[(atmprim, cco2, 'bcoeff')]
        asurf = all_coeffs[(atmprim, cco2, 'asurf')]
        bsurf = all_coeffs[(atmprim, cco2, 'bsurf')]

        h_ab = hr_from_ab(acoeff, bcoeff, asurf, bsurf, temp, surf_temp)
        #print(atm, xi, np.mean(h_ab))
        hr_somma += xi * h_ab[:n_alts]

    hr_somma = hr_somma/np.sum(xis)

    return hr_somma


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

    fu = np.zeros(len(allatms))
    for ialt in range(n_alts):
        fuialt = delta_xi_at_x0(xis, cco2, ialt)
        for i, atm in enumerate(allatms):
            fu[i] += fuialt[i]**2

    fu = np.sqrt(fu)

    return fu

def delta_xi_at_x0(xis, cco2, ialt, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths, squared_residuals = False):
    """
    This is done for a single altitude x0.
    The delta function at page 511 bottom. xis is the set of weights in the order of allatms.
    """

    fu = np.zeros(len(allatms))
    for i, atm in enumerate(allatms):
        hr = all_coeffs[(atm, cco2, 'hr_ref')][ialt]
        #hr_somma = hr_from_xi(xis, atm, cco2)[ialt]
        hr_somma = hr_from_xi_at_x0(xis, atm, cco2, ialt)

        if not squared_residuals:
            fu[i] += atmweigths[atm] * (hr_somma - hr)
        else:
            fu[i] += atmweigths[atm] * (hr_somma - hr)**2

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

    for i in range(len(allatms)):
        for k in range(len(xis)):
            #print(i,k)
            J[i,k] = 1/(delta[i]) * np.sum([alldeltas[ialt][i]*jacall[i,k,ialt] for ialt in range(n_alts)])

    #print(np.mean(J))
    return J


def jacdelta_xi_at_x0(xis, cco2, ialt, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths):
    """
    Jacobian of delta_xi_at_x0.
    """

    J = np.empty((len(allatms), len(xis)))

    for i in range(len(allatms)):
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


def jacdelta_xi_all_x0s_fast(xis, cco2, all_coeffs = all_coeffs, atm_pt = atm_pt, atmweigths = atmweigths):
    """
    Jacobian of delta_xi_at_x0.
    """

    J = np.empty((len(allatms), len(xis), n_alts))

    for i in range(len(allatms)):
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

def plot_pdfpages(filename, figs, save_single_figs = True, fig_names = None):
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


def manuel_plot(y, xs, labels, xlabel = None, ylabel = None, title = None, xlimdiff = None):
    """
    Plots plt.plot(x, y, lab) for each x in xs. Plots the differences of all xs wrt xs[0] in a side plot.
    """
    fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
    colors = color_set(len(xs))
    i = 0
    for x, lab, col in zip(xs, labels, colors):
        a0.plot(x, y, label = lab, color = col)
        if i == 0:
            i+=1
            continue
        a1.plot(x - xs[0], y, color = col)
        a1.axvline(0., color = 'grey', alpha = 0.6)

    a0.axhline(70., color = 'red', alpha = 0.6, linestyle = '--')
    a0.axhline(85., color = 'orange', alpha = 0.6, linestyle = '--')
    a1.axhline(70., color = 'red', alpha = 0.6, linestyle = '--')
    a1.axhline(85., color = 'orange', alpha = 0.6, linestyle = '--')
    a0.grid()
    a1.grid()
    if xlimdiff is not None:
        a1.set_xlim(xlimdiff)
    a0.legend(loc = 3)

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
