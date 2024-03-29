# LISTA del codice flow

Lista files vari:

- all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v4.p'))
-- Contiene gli a e b coeffs originali (prima di interpolare roba) per tutte le atm e i co2. Da ricalcolare per il settimo co2 prof. Prodotto da fomi_calc_ab_coeffs_LTE.py.

- atm_pt = pickle.load(open(cart_out + 'atm_pt_v4.p'))
-- Contiene i profili P,T delle atmosfere e i profili di co2. Da ricalcolare per il settimo co2 prof. Prodotto da fomi_calc_ab_coeffs_LTE.py.

- best_unif = pickle.load(open(cart_out+'best_uniform_allco2.p'))
- best_unif_v2 = pickle.load(open(cart_out+'best_uniform_allco2_v2.p'))
- best_var = pickle.load(open(cart_out+'best_atx0_allco2.p'))
- best_var.update(pickle.load(open(cart_out+'best_atx0_allco2_High.p')))
-- DEPRECATED. Risultati dei fit "a mano" prodotti da fomi_multiatmco2_ab_LTE_v2.py. Non servono più.

- tot_coeff_co2 = pickle.load(open(cart_out + 'tot_coeffs_co2_v1_LTE.p', 'r'))
-- DEPRECATED. Vecchi coeffs a e b per ogni co2 prodotti dal fit di fomi_ab_LTE_step5.py. Ora c'è la v2.

- tot_coeff_co2 = pickle.load(open(cart_out + 'tot_coeffs_co2_v2_LTE.p', 'r'))
-- Nuovi coeffs a e b per ogni co2 prodotti da check_LTE_fit_v2.py.

- varfit_xis = pickle.load(open(cart_out+'varfit_LTE_v2.p', 'rb'))
- varfit_xis_2 = pickle.load(open(cart_out+'varfit_LTE_v3.p', 'rb'))
-- Coeffs della nuova param LTE! Prodotti da fomi_multiatmco2_ab_LTE_v3.py. Da ricalcolare per il settimo co2 prof.

-------------------------------------------------------------------------------------------------------
---------------- Flow del codice attuale.

Per la param LTE:
1 - fomi_calc_ab_coeffs_LTE.py
2 - fomi_multiatmco2_ab_LTE_v4.py (was: fomi_multiatmco2_ab_LTE_v3.py) -> varfit4, varfit5
3 - check_LTE_fit_v2.py -> check_newparam_LTE_final_LEASTSQUARES_v3_abfit.pdf

Per la param NLTE low trans:
4 - trans_region_low.py -> all_coeffs_NLTE.p
5 - fomi_multiatmco2_ab_NLTE.py -> varfit4_nlte, varfit5_nlte
6 - check_NLTE_fit_low.py -> check_newparam_NLTE_lowtrans.pdf, produces tot_coeffs_co2_NLTE.p

Per la param della upper transition region e della cool-to-space region:
7 - fomi_recurformula_allatm.py

-----
Full atmosphere check:
8 - test_newparam_full.py


#################################################
----- Flow con nuova strategia.

LTE + low trans:

1 - reparam_weofs_v2.py -> produces the new set of coeffs for LTE (regression with 1st and 2nd eofs):
    -> regrcoef_v3.p : new set of LTE coefficients
    -> check_reparam_LTE.pdf : ci dev'essere un errore nel plot (spero)

2 - reparam_lowtrans_v3.py -> implementation of the LTE and NLTE low trans correction:
    -> nlte_corr_low.p : coeffs for nlte correction
    -> check_reparam_nLTE_low.pdf : check HR LTE + low trans
    -> check_reparam_NLTEcorrection.pdf : check of the NLTE low trans correction

high trans:

3 - reparam_high_v8fin.py -> implements the fit of the alpha correction for the recurrence formula:
    -> alpha_fit_high.p : regression coefficients for the pop_4e strategy (pop_nl0 not saved yet)
    -> check_alpha_popup_relerr_v8.pdf : check of the alpha fit with the different strategies
    -> check_reparam_high_v8.pdf : check HR high trans

full atm:

4 - test_reparam_full.py


########

Se cambi L_esc, devi rilanciare:

- fomi_recurformula_allatm (produce cose_upper_atm)
- reparam_high_v8fin, reparam_high_v9_inverse, reparam_high_v10 (cambia L_esc)
- check_fomialpha_refatm_v8_vs_v9: -> choose best param (cambia L_esc)
- test_reparam_full -> new set of coeffs_finale.p!

########################################################################
---------------- Flow del codice parte NLTE (se cambiano le ref calc) (27/09/2022)

Per la param LTE:
1 - fomi_calc_ab_coeffs_LTE.py
2 - fomi_multiatmco2_ab_LTE_v4.py (was: fomi_multiatmco2_ab_LTE_v3.py) -> varfit4, varfit5
3 - check_LTE_fit_v2.py -> check_newparam_LTE_final_LEASTSQUARES_v3_abfit.pdf

Per la param NLTE low trans:
4 - trans_region_low.py -> all_coeffs_NLTE.p
5 - fomi_multiatmco2_ab_NLTE.py -> varfit4_nlte, varfit5_nlte
6 - check_NLTE_fit_low.py -> check_newparam_NLTE_lowtrans.pdf, produces tot_coeffs_co2_NLTE.p

Per la param della upper transition region e della cool-to-space region:
7 - reparam_high_v12.py -> calculates alpha unif
8 - reparam_high_v11start.py -> calculates alpha unif starting from ref
9 - test_newparam_full.py -> produces coeffs_finale.p!!

Per il confronto con MIPAS:
10 - MIPAS_cool_final.py -> produces all mipcalc calculations and plots
11 - granada_finalplots.py -> refcalc, plot of weights for a, b and alpha
