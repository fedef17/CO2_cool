# CO2_cool
New parametrization of CO2 heating rate in 15um band in non-LTE conditions.

## To install:
- Install python environment in file environment.yml (with conda: `conda env create -f environment.yml`)

## To use:
- Use run_new_param.py script. Atmospheric input needed in the input.dat format. Run as: `python run_new_param.py my_input_file`. Output is saved in output.dat.

- Alternatively, use `import new_param_lib_light as npl` in a Python script or notebook. You can then use the `npl.new_param_full_allgrids_v1()` directly in the script.

- Inputs needed: surface temperature, vertical profiles of pressure, temperature, VMRs of CO2, O, O2, N2. Profiles from ground to top, temperature in K, pressure in hPa, vmrs in absolute fraction (not ppm). 

- Profiles are needed up to 0.001 hPa for the parametrization to work. Lower level should be at the ground for an accurate calculation.

- Surface temperature can be specified in the command line using the argument `--surf_temp VALUE`, e.g. `python run_new_param.py my_input_file --surf_temp 290.6`. If not given, the run script assumes that the first level in the temperature profile is equal to the surface temperature (surf_temp = temp[0]).

- Output file name can be specified via the `--output my_out_file` argument.

## To test:
- After installation, run: `python run_new_param.py input.dat`. The output in output.dat should be the same as in the output_test.dat.
- You can find more input/output test atmospheres for various conditions in the test/ folder.

## To modify the collisional rates:
'a_zo' : 3.5e-13, 'b_zo' : 2.32e-9, 'g_zo' : 76.75, 'a_zn2' : 7e-17, 'b_zn2' : 6.7e-10, 'g_zn2' : 83.8, 'a_zo2' : 7e-17, 'b_zo2' : 1.0e-9, 'g_zo2' : 83.8
- Rates are defined in the form: z = a*np.sqrt(T) + b * np.exp(-g * T**(-1./3)). Name of the coefficients are as follows: 
    - for CO2-O: a_zo, b_zo, g_zo  (default: 3.5e-13, 2.32e-9, 76.75)
    - for CO2-O2: a_zo2, b_zo2, g_zo2  (default: 7.0e-17, 1.0e-9, 83.8)
    - for CO2-N2: a_zn2, b_zn2, g_zn2  (default: 7.0e-17, 6.7e-10, 83.8)

- You can specify different values for some or all these coeffs in the mod_rates.yml file. Then, to use the new rates, run the script with the '--mod_rates' option, like `python run_new_param.py input.dat --mod_rates`. If using the function directly, set the coeffs as keyword arguments in the call to npl.new_param_full_allgrids_v1() (e.g. "a_zo = 1.75e-13").



