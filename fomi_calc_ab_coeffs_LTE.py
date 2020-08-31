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

if os.uname()[1] == 'ff-clevo':
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/SpectRobot/')
    sys.path.insert(0, '/home/fedefab/Scrivania/Research/Post-doc/git/pythall/')
    cart_out = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/LTE/'
    cartsav = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/sent2/sav/'
    cartatm = '/home/fedefab/Scrivania/Research/Post-doc/CO2_cooling/new_param/sent2/atm/'
elif os.uname()[1] == 'hobbes':
    sys.path.insert(0, '/home/fabiano/Research/git/SpectRobot/')
    sys.path.insert(0, '/home/fabiano/Research/git/pythall/')
    cart_out = '/home/fabiano/Research/lavori/CO2_cooling/new_param/LTE/'
    cartatm = '/home/fabiano/Research/lavori/CO2_cooling/new_param/sent6_cm_v3/atm/cira_v3/'
    cartsav = '/home/fabiano/Research/lavori/CO2_cooling/new_param/sent6_cm_v3/sav_v3/'

import spect_base_module as sbm
import spect_classes as spcl

kbc = const.k/(const.h*100*const.c) # 0.69503
kboltz = 1.38064853e-23 # J/K

cp = 1.005e7 # specific enthalpy dry air - erg g-1 K-1
#############################################################
###################################################################

filsav = 'data_cira_{}_co2_{}.sav'
filvmr = 'vmr_cira_{}_co2_{}.prf'
filatm = 'pt_cira_{}.prf'

frini = 540.
frfin = 800.
all_freqs = np.arange(frini, frfin+1, 10.)
spect_ranges = [(fr1, fr2) for fr1, fr2 in zip(all_freqs[:-1], all_freqs[1:])]
numid = np.array([np.mean([fr1, fr2]) for fr1, fr2 in spect_ranges])

fun = '0 0 0 0'
v1 = '0 1 1 0'
v2 = '0 2 2 0'
v3 = '0 3 3 0'
#[('1 1 1 0', '1 0 0 0')]:#

all_trans = [(v1, fun), (v2, v1), (v3, v2)]
all_temps = np.arange(140, 321, 5)
all_total_strengths = pickle.load(open(cart_out + 'line_strengths_CO2.p'))

##############################################################################

atm = 'mle'
cco2 = 1
allatms = ['mle', 'mls', 'mlw', 'tro', 'sas', 'saw']
allco2 = np.arange(1,7)

all_coeffs = dict()
atm_pt = dict()

for atm in allatms:
    for cco2 in allco2:
        print(atm, cco2)
        atmalts, atmtemp, atmpres = sbm.read_input_atm_man(cartatm+filatm.format(atm))
        atmalts, mol_vmr, _, _ = sbm.read_input_vmr_man(cartatm+filvmr.format(atm, cco2))

        coso = io.readsav(cartsav+filsav.format(atm, cco2))['data']
        nomi = coso.dtype.names
        #('WNUMLOW', 'WNUMHIGH','CM', 'CSURF', 'LSPACE', 'HR_KDAY', 'SRCFCTN', 'RSURF', 'LOG_PRESS', 'TEMPERATURE', 'PRESSURE', 'ALTITUDE', 'CO2_VMR', 'O_VMR')

        cm = coso.CM[0]
        hr = coso.HR_KDAY[0]
        lspace = coso.LSPACE[0]
        csurf = coso.CSURF[0]

        co2 = coso.CO2_VMR[0]
        oat = coso.O_VMR[0]
        alts = coso.ALTITUDE[0]
        temp = coso.TEMPERATURE[0]
        pres = coso.PRESSURE[0]

        n_dens = sbm.num_density(pres, temp)

        atm_pt[(atm, 'temp')] = temp
        atm_pt[(atm, 'surf_temp')] = atmtemp[0]
        atm_pt[(atm, 'pres')] = pres
        atm_pt[(atm, 'alts')] = alts
        atm_pt[(atm, 'ndens')] = n_dens
        atm_pt[(atm, cco2, 'co2')] = co2
        atm_pt[(atm, cco2, 'oat')] = oat

        frlo = coso.WNUMLOW[0]
        frhi = coso.WNUMHIGH[0]
        #
        # # quindi abbiamo 66 alts, 26 freq intervals
        # # ora bisogna capire come usare ste cose
        # # ...
        n_alts = len(alts)
        n_sp = len(numid)

        epsilon = np.zeros((n_alts, n_sp),  dtype = float)
        acoeff = np.zeros((n_alts, n_alts, n_sp),  dtype = float)
        bcoeff = np.zeros((n_alts, n_alts, n_sp),  dtype = float)
        asurf = np.zeros((n_alts, n_sp),  dtype = float)
        bsurf = np.zeros((n_alts, n_sp),  dtype = float)

        # QUi EVENTUALMENTE AGGIUNGI ccoeff

        #cm = cm[::-1, ::-1, :]
        #cm = cm[:, :, ::-1]
        #dens = sbm.air_density(pres, temp, MM = 29)

        for s in range(n_sp):
            # phi = np.exp(-numid[s]/(kbc*temp))
            # phi_g = np.exp(-numid[s]/(kbc*atmtemp[0]))
            phi_BB = np.array([spcl.Calc_BB_single(numid[s], T) for T in temp])
            phi_BB_g = spcl.Calc_BB_single(numid[s], atmtemp[0])

            for xi in range(n_alts):
                epsilon[xi, s] = np.sum(cm[:, xi, s] * phi_BB) # il contributo della colonna
                epsilon[xi,s] += phi_BB_g * csurf[xi, s] # il contributo del ground!
                #epsilon[xi,s] -= phi[xi] * lspace[xi, s] # il contributo del cool to space! No, questo è già implicito. La CM tiene conto del riscaldamento dello strato da parte degli altri strati. Non ha senso parlare qui di cool to space. Quello che devo considerare è invece un eventuale heating solare, ma in questa banda non conta.

            epsilon[:,s] = epsilon[:,s]/ ( n_dens * (2.5 * oat + 3.5 * (1 - oat ) ) )

        fac = 86400*1.e-7 / kboltz
        epsilon = epsilon * fac

        eps_sum = np.array([np.sum([epsilon[xi,s] for s in range(n_sp)]) for xi in range(n_alts)])
        hr_sum = np.array([np.sum([hr[xi,s] for s in range(n_sp)]) for xi in range(n_alts)])

        A_cm = np.empty_like(cm)
        A_surf = np.empty_like(csurf)

        # quindi A = fac * CM / (n_dens * (2.5 * oat + 3.5 * (1 - oat ))) * phi_BB / phi
        for s in range(n_sp):
            phi = np.exp(-numid[s]/(kbc*temp))
            phi_BB = np.array([spcl.Calc_BB_single(numid[s], T) for T in temp])
            phi_g = np.exp(-numid[s]/(kbc*atmtemp[0]))
            phi_BB_g = spcl.Calc_BB_single(numid[s], atmtemp[0])

            A_surf[:, s] = fac * csurf[:, s] / (n_dens * (2.5 * oat + 3.5 * (1 - oat ))) * phi_BB_g / phi_g
            for xi in range(n_alts):
                A_cm[:, xi, s] = cm[:, xi, s] * phi_BB / phi

            for xi in range(n_alts):
                A_cm[xi, :, s] = A_cm[xi, :, s] * fac / (n_dens * (2.5 * oat + 3.5 * (1 - oat )))

            #     epsilon[xi, s] = np.sum(cm[:, xi, s] * phi_BB) # il contributo della colonna
            #     epsilon[xi,s] += phi_BB_g * csurf[xi, s] # il contributo del ground!
            # epsilon[:,s] = epsilon[:,s]/ ( n_dens * (2.5 * oat + 3.5 * (1 - oat ) ) )


        epsilon_A = np.empty_like(epsilon)
        for s in range(n_sp):
            phi = np.exp(-numid[s]/(kbc*temp))
            phi_g = np.exp(-numid[s]/(kbc*atmtemp[0]))

            for xi in range(n_alts):
                epsilon_A[xi, s] = np.sum(A_cm[:, xi, s] * phi) # il contributo della colonna
                epsilon_A[xi,s] += phi_g * A_surf[xi, s] # il contributo del ground!

        eps_sum = np.sum(epsilon_A, axis = 1)

        # plt.ion()
        # fig = plt.figure()
        # plt.plot(hr_sum, alts, label = 'reference hr')
        # plt.plot(eps_sum, alts, label = 'hr from cm')
        # plt.plot(eps_sum-hr_sum, alts, label = 'diff')
        # plt.legend()
        # plt.xlabel('HR (K/day)')
        # plt.ylabel('Altitude (km)')
        # fig.savefig(cart_out + 'hr_from_cm_compare_Acm.pdf')
        #
        # plt.figure()
        # plt.plot(eps_sum/hr_sum, alts, label = 'ratio')
        # plt.xlabel('ratio of calculated to reference hr')
        # plt.ylabel('Altitude (km)')
        # fig.savefig(cart_out + 'hr_from_cm_ratio_Acm.pdf')

        # STEP 2 pagina 511
        all_total_strengths, all_hitran_strengths = pickle.load(open(cart_out + 'line_strengths_CO2.p'))

        for s in range(n_sp):
            S0 = all_hitran_strengths[('fun', spect_ranges[s], 'all')] # all isotopes
            S1 = all_hitran_strengths[('1st hot', spect_ranges[s], 'all')] # all isotopes
            S2 = all_hitran_strengths[('2nd hot', spect_ranges[s], 'all')] # all isotopes
            phi = np.exp(-numid[s]/(kbc*temp))
            for xi in range(n_alts):
                acoeff[:, xi, s] = A_cm[:, xi, s] * S0 / (S0 + (S1+S2)*phi[xi])
                bcoeff[:, xi, s] = A_cm[:, xi, s] * (S1+S2) / (S0 + (S1+S2)*phi[xi])

                asurf[xi, s] = A_surf[xi, s] * S0 / (S0 + (S1+S2)*phi[xi])
                bsurf[xi, s] = A_surf[xi, s] * (S1+S2) / (S0 + (S1+S2)*phi[xi])

            # Questo sarebbe più corretto ma non è così
            #phi_g = np.exp(-numid[s]/(kbc*atmtemp[0]))
            #asurf[:, s] = A_surf[:, s] * S0 / (S0 + (S1+S2)*phi_g)
            #bsurf[:, s] = A_surf[:, s] * (S1+S2) / (S0 + (S1+S2)*phi_g)

        # STEP 3 pagina 511: facciamo la media spettrale
        acoeff_tot = np.zeros((n_alts, n_alts), dtype = float)
        bcoeff_tot = np.zeros((n_alts, n_alts), dtype = float)
        asurf_tot = np.zeros(n_alts, dtype = float)
        bsurf_tot = np.zeros(n_alts, dtype = float)

        E_fun = 667.3799 # cm-1 energy of the 0110 -> 0000 transition

        for s in range(n_sp):
            phi = np.exp(-numid[s]/(kbc*temp))
            phi_fun = np.exp(-E_fun/(kbc*temp))
            phi_g = np.exp(-numid[s]/(kbc*atmtemp[0]))
            phi_fun_g = np.exp(-E_fun/(kbc*atmtemp[0]))
            for xi in range(n_alts):
                acoeff_tot[:, xi] += acoeff[:, xi, s] * phi / phi_fun
                bcoeff_tot[:, xi] += bcoeff[:, xi, s] * (phi * phi[xi]) / (phi_fun * phi_fun[xi])

                asurf_tot[xi] += asurf[xi, s] * phi_g / phi_fun_g
                bsurf_tot[xi] += bsurf[xi, s] * (phi_g * phi[xi]) / (phi_fun_g * phi_fun[xi])

        epsilon_ab_tot = np.zeros(len(alts), dtype = float)

        phi_fun = np.exp(-E_fun/(kbc*temp))
        phi_fun_g = np.exp(-E_fun/(kbc*atmtemp[0]))

        # THIS IS THE FINAL FORMULA FOR RECONSTRUCTING EPSILON FROM a AND b
        for xi in range(n_alts):
            epsilon_ab_tot[xi] = np.sum((acoeff_tot[:, xi] + bcoeff_tot[:, xi]* phi_fun[xi]) * phi_fun) # il contributo della colonna
            epsilon_ab_tot[xi] += (asurf_tot[xi] + bsurf_tot[xi]* phi_fun[xi]) * phi_fun_g # il contributo del ground!

        # plt.ion()
        # fig = plt.figure()
        # plt.plot(hr_sum, alts, label = 'reference hr')
        # plt.plot(epsilon_ab_tot, alts, label = 'hr from cm')
        # plt.plot(epsilon_ab_tot-hr_sum, alts, label = 'diff')
        # plt.legend()
        # plt.xlabel('HR (K/day)')
        # plt.ylabel('Altitude (km)')
        # fig.savefig(cart_out + 'hr_from_cm_compare_abcalc.pdf')
        #
        # plt.figure()
        # plt.plot(epsilon_ab_tot/hr_sum, alts, label = 'ratio')
        # plt.xlabel('ratio of calculated to reference hr')
        # plt.ylabel('Altitude (km)')
        # fig.savefig(cart_out + 'hr_from_cm_ratio_abcalc.pdf')

        # salvo gli a e i b per questa atmosfera
        all_coeffs[(atm, cco2, 'acoeff')] = acoeff_tot
        all_coeffs[(atm, cco2, 'bcoeff')] = bcoeff_tot
        all_coeffs[(atm, cco2, 'asurf')] = asurf_tot
        all_coeffs[(atm, cco2, 'bsurf')] = bsurf_tot
        all_coeffs[(atm, cco2, 'A_cm')] = A_cm
        all_coeffs[(atm, cco2, 'A_surf')] = A_surf
        all_coeffs[(atm, cco2, 'hr_ref')] = hr_sum

pickle.dump(all_coeffs, open(cart_out + 'all_coeffs_LTE_v2.p', 'w'))
pickle.dump(atm_pt, open(cart_out + 'atm_pt_v2.p', 'w'))

# STEP 4 pagina 511: facciamo la media pesata tra le diverse atmosfere
# new_file
