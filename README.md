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

For more details, refer to `docs/manual.stub` (in progress...)

---

## List of published articles and theses using PyXover ##

<a id="1"></a>1. **Desprats W., S. Bertone, D. Arnold, et al.** (2024). *Combination of altimetry crossovers and Doppler observables for orbit determination and geodetic parameter recovery: application to Callisto*. Accepted by Acta Astronautica. [10.1016/j.actaastro.2024.10.045](https://doi.org/10.1016/j.actaastro.2024.10.045)  

<a id="2"></a>2. **Grisolia, M.** (2024). *Validation of radioscience derived orbits by crossovers analyses of the Mercury Laser Altimeter*. B.Sc. Thesis, Polytechnic University of Turin (Italy).

<a id="3"></a>3. **Desprats, W.** (2024). *Callisto geodesy: A simulation study to support further space missions to the Jovian system*. Ph.D. Thesis, Astronomical Institute, University of Bern (Switzerland) 

[//]: # (Available at: [link]&#40;https://example.com&#41;  )
<a id="4"></a>4. **Bertone S., E. Mazarico, M. K. Barker, et al.** (2021). *Deriving Mercury geodetic parameters with altimetric crossovers from the Mercury Laser Altimeter (MLA)*. Journal of Geophysical Research - Planets, **126**(4): e2020JE006683. [10.1029/2020JE006683](http://dx.doi.org/10.1029/2020JE006683)

<a id="5"></a>5. **Hosseiniarani, A.** (2020). *BepiColombo Laser Altimeter (BELA) Performance Evaluation: From Laboratory Tests to Simulations of Flight Observations*. Ph.D. Thesis, Space Research & Planetary Sciences, University of Bern (Switzerland) 
