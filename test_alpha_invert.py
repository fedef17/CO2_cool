def calc_alpha2(cco2, atm, force_min_alpha = 0.5):
    print(cco2, atm)

    n_alts_cs = 80

    hr_ref = all_coeffs_nlte[(atm, cco2, 'hr_ref')]
    L_esc = cose_upper_atm[(atm, cco2, name_escape_fun)]
    lamb = cose_upper_atm[(atm, cco2, 'lamb')]
    co2vmr = cose_upper_atm[(atm, cco2, 'co2vmr')]
    #MM = cose_upper_atm[(atm, cco2, 'MM')]
    temp = atm_pt[(atm, 'temp')]

    ovmr = all_coeffs_nlte[(atm, cco2, 'o_vmr')]
    o2vmr = all_coeffs_nlte[(atm, cco2, 'o2_vmr')]
    n2vmr = all_coeffs_nlte[(atm, cco2, 'n2_vmr')]

    MM = npl.calc_MM(ovmr, o2vmr, n2vmr)
    #lamb = npl.calc_lamb(pres, temp, ovmr, o2vmr, n2vmr)

    hr_calc = npl.hr_reparam_low(cco2, temp, surf_temp, regrcoef = regrcoef, nlte_corr = nlte_corr)

    hr_ref_cal = hr_ref.copy()
    hr_ref_cal[:alt2] = hr_calc[:alt2]

    alpha_top = dict()
    hr_tops = dict()

    allntops = [55, 60, 65, 70, 75]
    for n_top in allntops:
        print(n_top)
        alpha_ref = npl.recformula_invert(hr_ref, L_esc, lamb, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, ovmr = ovmr, force_min_alpha = force_min_alpha)
        alpha_top[n_top] = alpha_ref

        print(alpha_ref)#[alt2-1:n_top])

        hr_calc_top = npl.recformula(alpha_ref, L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, n_alts_cs = n_alts_cs, ovmr = ovmr)
        hr_tops[n_top] = hr_calc_top

    hr_calc1 = npl.recformula(np.ones(len(alpha_ref)), L_esc, lamb, hr_calc, co2vmr, MM, temp, n_alts_trlo = alt2, n_alts_trhi = n_top, n_alts_cs = n_alts_cs, ovmr = ovmr)
    labels = ['hr_ref']+['top {}'.format(nto) for nto in allntops] + ['alpha1']
    hrs = [hr_ref] + [hr_tops[nto] for nto in allntops] + [hr_calc1]

    # labels = ['hr_ref']+['top {}'.format(nto) for nto in allntops]
    # hrs = [hr_ref] + [hr_tops[nto] for nto in allntops]

    colors = npl.color_set(7)
    tit = 'cco2 {} - atm {}'.format(cco2, atm)
    fig, a0, a1 = npl.manuel_plot(np.arange(npl.n_alts_all), hrs, labels, xlabel = xlab, ylabel = ylab, title = tit, xlimdiff = (-20, 20), xlim = (-1000, 10), ylim = (40, 80), linestyles = ['-', ':', ':', ':', ':', ':', ':'], colors = colors, orizlines = [40, alt2, n_top], linewidth = 2.)

    fig2 = plt.figure()
    for nto, col in zip(allntops, colors[1:]):
        plt.plot(alpha_top[nto], np.arange(alt2-1, nto), color = col)
        plt.grid()
        plt.gca().set_xscale('log')
        plt.gca().set_xlim([0.5, 1000.])

    return
