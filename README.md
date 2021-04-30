# pyxover - Altimetry Analysis Tools For Planetary Geodesy

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
 headaches and results need to be thoroughly checked and taken with an healthy degree of mistrust!
A major clean-up and merging with concurrent local versions is pending, waiting for funding approval.
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

!! Cartopy (`https://scitools.org.uk/cartopy/docs/latest/index.html`) and GDAL 
(`https://gdal.org`) libraries might be required by some routines 
(especially plotting and evaluation tools) and they need to be installed 
separately !!

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
