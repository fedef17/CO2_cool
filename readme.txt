# LISTA del codice flow

Lista files vari:

- all_coeffs = pickle.load(open(cart_out + 'all_coeffs_LTE_v2.p'))
-- Contiene gli a e b coeffs originali (prima di interpolare roba) per tutte le atm e i co2. Da ricalcolare per il settimo co2 prof. Prodotto da fomi_calc_ab_coeffs_LTE.py.

- atm_pt = pickle.load(open(cart_out + 'atm_pt_v2.p'))
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
6 - check_NLTE_fit_low.py -> check_newparam_NLTE_lowtrans.pdf

Per la param della cool-to-space region:
7 - fomi_recurformula.py
