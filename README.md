# pyxover - Altimetry Analysis Tools For Planetary Geodesy
[![DOI](https://zenodo.org/badge/319753409.svg)](https://zenodo.org/badge/latestdoi/319753409)

Software suite for the analysis of laser altimetry data: from standard range data (PDS RDR) to planetary orientation and
 tides. Existing or simulated ranges (pyaltsim) are geolocated on the planetary surface
 on the basis of input emitter trajectory and planetary rotation model (pygeoloc).
 Intersections between such groundtrack are located by a computationally efficient
 multi-threaded algorithm: elevation differences, or discrepancies, are computed toghether with 
 numerical partial derivatives with respect to the chosen solve-for parameters (pyxover).
 Crossovers residuals and partial derivatives contribute to the equations system solved by 
 least-squares in ``accumxov''. Statistics and plotting tools are also provided (xovutil).
 
## Disclaimer

This is scientific code in ongoing development: using it might be tricky, reading it can cause 
 headaches and results need to be thoroughly checked and taken with a healthy degree of mistrust!
Use it at your own risk and responsibility. 
 ## Installation ##

### Set up a virtual environment and clone the repository ###

Make a new directory and clone this repository to it. Then, inside the
directory that you've just created, run `python -m venv env`. This will
create a "virtual environment", which is a useful tool that Python
provides for managing dependencies for projects. The new directory
"contains" the virtual environment.

### Activate the virtual environment ###

To activate the virtual environment, run `source env/bin/activate` from
the directory containing it. Make sure to do this before doing
anything else below.

### Getting the dependencies ###

Install the rest of the dependencies by running `pip install -r
requirements.txt`.

[//]: # (!! Cartopy &#40;`https://scitools.org.uk/cartopy/docs/latest/index.html`&#41; and GDAL )

[//]: # (&#40;`https://gdal.org`&#41; libraries might be required by some routines )

[//]: # (&#40;especially plotting and evaluation tools&#41; and they need to be installed )

[//]: # (separately !!)

### Installing this package ###

Finally, run:
``` shell
pip install .
```
To install the package in editable mode, so that changes in this
directory propagate immediately, run:
``` shell
pip install -e .
```
To test the installation, from the project directory, run:
``` shell
python setup.py test
```

## Running the examples ##

The examples directory contains the setup to process altimetry ranges by the Mercury
 Laser Altimeter (MLA) onboard the MESSENGER mission, illustrating how
 to use this package. To run the example, you'll need to import the required spice
 kernels listed in `examples/MLA/data/aux/mymeta` to `examples/MLA/data/aux/kernels/` and eventually adapt `mymeta`.

 Then, try:
``` shell
cd examples
python mla_iter.py
```
Else, check out the `tests` directory for a "simpler" approach.

For more details, refer to `docs/manual.stub` (in progress...) and the Configuration reference below for the most common runtime options.

## Configuration reference

Most processing options are configured through `XovOpt` in `src/config.py`. Key fields include:

| Field | Default | Notes |
| --- | --- | --- |
| `body` / `instrument` | `MERCURY` / `MLA` | Central body and instrument identifiers used by SPICE kernels. |
| `basedir` (and derived `rawdir`, `outdir`, `auxdir`, `tmpdir`) | `pawstel/data/` | Base directory for inputs/outputs; dependent paths are recomputed during consistency checks. |
| `n_proc` | `mp.cpu_count() - 3` | Number of worker processes. |
| `expopt` | `BS0` | Experiment label used to name input/output folders. |
| `parOrb`, `parGlo` | See defaults in `src/config.py` | Initial perturbations for orbital and global parameters. |
| `par_constr`, `mean_constr` | See defaults | Constraints applied during least-squares solutions. |
| `cloop_sim`, `pert_cloop_orb`, `pert_cloop_glo`, `pert_tracks` | `False` / `{}` / `{}` / `[]` | Enable and tune closed-loop simulations or per-track perturbations. |
| `sol4_orb`, `sol4_orbpar`, `sol4_glo` | `[None]`, `[None]`, `['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL']` | Select which orbital/global parameters are solved. |
| `OrbRep` | `cnt` | Orbital representation (`cnt`, `lin`, or `quad`). |
| `SpInterp`, `spice_meta`, `spice_spk` | `0`, `mymeta`, `[]` | SPICE interpolation settings and additional kernels. |
| `new_gtrack`, `new_xov` | `2`, `2` | Whether to regenerate geolocated tracks or crossovers (`0` skip, `1` create if missing, `2` recreate). |
| `import_proj`, `import_abmat` | `False`, `""` | Import precomputed projections or A/B matrices instead of recomputing. |
| `weekly_sets`, `monthly_sets`, `multi_xov`, `new_algo`, `compute_input_xov` | `False`, `False`, `False`, `True`, `True` | Various runtime toggles for crossover processing. |
| `msrm_sampl` | `2` | Measurement downsampling factor; **must be an even integer**, otherwise configuration validation fails. |
| `n_interp` | `6` | Number of laser altimetry points on either side of a crossover used for interpolation. |
| `full_covar`, `roughn_map` | `False`, `False` | Control covariance computation and roughness map usage. |
| `new_illumNG`, `apply_topo`, `small_scale_topo` | `True`, `True`, `False` | PyAltSim options for illumination and topography handling. |
| `range_noise`, `range_noise_mean_std` | `True`, `[0., 0.]` | Toggle and characterize simulated range noise. |
| `local_dem`, `max_range_altitude`, `sampling_rate` | `True`, `1050`, `10` | DEM usage, maximum range altitude (km), and sampling rate (Hz). |
| `vecopts` | See defaults | SPICE identifiers and frame names for the spacecraft and planet. |

---

## List of published articles and theses using PyXover ##

<a id="1"></a>1. **Desprats W., S. Bertone, D. Arnold, et al.** (2024). *Combination of altimetry crossovers and Doppler observables for orbit determination and geodetic parameter recovery: application to Callisto*. Accepted by Acta Astronautica. [10.1016/j.actaastro.2024.10.045](https://doi.org/10.1016/j.actaastro.2024.10.045)  

<a id="2"></a>2. **Grisolia, M.** (2024). *Validation of radioscience derived orbits by crossovers analyses of the Mercury Laser Altimeter*. B.Sc. Thesis, Polytechnic University of Turin (Italy).

<a id="3"></a>3. **Desprats, W.** (2024). *Simulation Study for Geodetic Parameter Recovery at Europa and Callisto*. Ph.D. Thesis, Astronomical Institute, University of Bern (Switzerland) 

[//]: # (Available at: [link]&#40;https://example.com&#41;  )
<a id="4"></a>4. **Bertone S., E. Mazarico, M. K. Barker, et al.** (2021). *Deriving Mercury geodetic parameters with altimetric crossovers from the Mercury Laser Altimeter (MLA)*. Journal of Geophysical Research - Planets, **126**(4): e2020JE006683. [10.1029/2020JE006683](http://dx.doi.org/10.1029/2020JE006683)

<a id="5"></a>5. **Hosseiniarani, A.** (2020). *BepiColombo Laser Altimeter (BELA) Performance Evaluation: From Laboratory Tests to Simulations of Flight Observations*. Ph.D. Thesis, Space Research & Planetary Sciences, University of Bern (Switzerland) 
