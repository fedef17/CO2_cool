def calc_alpha2(cco2, atm):
    print(cco2, atm)
    hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_ref')]
    L_esc = cose_upper_atm[(atm, cco2, name_escape_fun)]
    lamb = cose_upper_atm[(atm, cco2, 'lamb')]
    co2vmr = cose_upper_atm[(atm, cco2, 'co2vmr')]
    MM = cose_upper_atm[(atm, cco2, 'MM')]
    temp = atm_pt[(atm, 'temp')]

    hr_calc = npl.hr_reparam_low(cco2, temp, surf_temp, regrcoef = regrcoef, nlte_corr = nlte_corr)

    hr_ref_cal = hr_ref.copy()
    hr_ref_cal[:alt2] = hr_calc[:alt2]

    alpha_ref = npl.recformula_invert(hr_ref, L_esc, lamb, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top)
    print(alpha_ref[alt2-1:n_top])
    #alpha_calc = npl.recformula_invert(hr_ref_cal, L_esc, lamb, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top)

    hr_calc_fref = npl.recformula(alpha_ref[alt2-1:n_top], L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top)

    hr_ref_fref = npl.recformula(alpha_ref[alt2-1:n_top], L_esc, lamb, hr_ref, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top)
    #hr_calc_fcalc = npl.recformula(alpha_calc[alt2:n_top+1], L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top)

    fig = plt.figure()
    #plt.plot(hr_ref_cal, np.arange(len(hr_ref)), label = 'ref_calc', ls = '--')
    plt.plot(hr_ref, np.arange(len(hr_ref)), label = 'ref')
    plt.plot(hr_calc_fref, np.arange(len(hr_ref)), label = 'alpha calc', ls = '--')
    plt.plot(hr_ref_fref, np.arange(len(hr_ref)), label = 'alpha ref', ls = ':')
    #plt.plot(hr_calc_fcalc, np.arange(len(hr_ref)), label = 'alpha calc', ls = '-.')
    plt.legend()
    plt.grid()
    plt.axhline(alt2, color = 'grey', ls = '-.')
    plt.axhline(n_top+1, color = 'grey', ls = '-.')
    plt.ylim(alt2-5, n_top + 10)
    plt.xlim(np.min(hr_ref[alt2:n_top+1])-5, 5)

    return hr_calc_fref, hr_calc_fcalc
