# CO2_cool
New parametrization of CO2 heating rate in 15um band in non-LTE conditions.

## To install:
- Install python environment in file environment.yml (with conda: `conda env create -f environment.yml`)

## To use:
- Use run_new_param.py script. Atmospheric input needed in the input.dat format. Run as: `python run_new_param.py`. Output is saved in output.dat.

- Alternatively, use `import new_param_lib_light as npl` in a Python script or notebook. You can then use the `npl.new_param_full_allgrids_v1()` directly in the script. Inputs needed: temperature, pressure, surface temperature, VMRs of CO2, O, O2, N2. First level at the ground, temperature in K, pressure in hPa, vmrs in absolute fraction (not ppm).

