#!/usr/bin/env python3
# ----------------------------------
# Read geolocalised MLA observations (gtrack), interpolate with gmt, then hillshade and plot
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#
import glob
import subprocess

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from prOpt import auxdir, tmpdir, outdir
from lib.plot_hillshade import plot_multiple


def import_dem(filein):
    # # grid point lists
    # from hillshade import hill_shade


    # open netCDF file
    dem_xarr = xr.open_dataset(filein)

    fig, ax1 = plt.subplots(1, 1)

    # converts dataset to dataarray
    dem_xarray = dem_xarr.to_array()

    # generates the rgb hillshaded intensity map (dem.data is a list of 2D arrays, hence the dem.data[0], azimuth=0 is left, then clockwise)
    # rgb = hill_shade(dem.data[0], elevation=10,azimuth=270,terrain=dem.data[0]*1.e3)

    # # rgb = np.linalg.norm(rgb,axis=2)
    # print(np.max(rgb,axis=1))
    # plt.imshow(rgb[700:1000,1500:1900]) #, cmap='gray', interpolation='bicubic')
    # # dem.plot.imshow()
    #
    # fig.savefig(
    #     filein.split('.')[0]+'.png',dpi=2000)
    return dem_xarray


def concat_elevations(files):
    dflist = []
    for f in files:
        data = pd.read_pickle(f).ladata_df
        if len(data) > 0:
            dflist.append(data[['LON', 'LAT', 'R']])
    dflist = pd.concat(dflist).round(2)
    dflist.to_csv(tmpdir + csvfile, sep='\t', index=False, header=False)

    return True

if __name__ == '__main__':

    ## GMT equivalent
    ## save defaults for Mercury in local dir
    # defaults - D > gmt.conf
    ## NP project:
    # gmt mapproject ladata_concat.txt -Js0/90/1:100000 -R0/360/75/90 -S -C -V > proj.txt
    ## compute median per bin:
    #  gmt blockmedian proj.txt -C -r -I.125 -R-1000/1000/-1000/1000 > proj_0125.txt
    ## interpolate:
    # surface -T0.75 -r -I.125 -R200/300/-400/-300 -Goutfile.grd
    ## subtract
    # grdmath

    sols = ['AGTP_0'] # tp4_0'] #,'KX1r2_17']
    subexp = '0res_1amp'
    # topomaps = []

    for sol in sols:
        csvfile = "ladata_concat_" + sol + ".txt"
        topofile = "topo_" + sol + ".grd"

        # import geolocalised elevations to single csv
        # ============================================
        # if True:
        files = glob.glob(outdir + "sim/" + sol + "/" + subexp + "/gtrack_*/gtrack_*.pkl")
        #concat_elevations(files)
        print("Done read+concat")

        # generate grd map
        # ================
        # print('mapproject', csvfile,'-Js0/90/1:1000 -R0/360/75/90 -S -C -bo | blockmedian -bi3 -C -r -I.05 -R-630/630/-630/630 -bo3 | surface -bi3 -T0.75 -r -I.05 -R-630/630/-630/630 -Gtopo_tp4_0.grd')
        command = ["mapproject "+csvfile+" -Js0/90/1:100000 -R0/360/60/90 -S -C -bo | "
                                         "blockmedian -bi3 -C -r -I1 -R-1500/1500/-1500/1500 -bo3 | "
                                         "surface -bi3 -T0.25 -r -I1 -R-1500/1500/-1500/1500 -G"+topofile]
        #r_dem = subprocess.check_output(command, universal_newlines=True, shell=True, cwd=tmpdir)
        print("Done generate grd")

        # import and plot DEM interpolated from residuals
        # ===============================================
        dem_xarray = import_dem(tmpdir+topofile)

        xrange = [700,1000]
        yrange = [1500,1900]
        data = dem_xarray.data[0][xrange[0]:xrange[1],yrange[0]:yrange[1]]

        plot_multiple(data)
        print("Done plotting topo")

    # exit()

    # # compare grd files and plot differences
    # # ======================================
    # dem_A = import_dem('plot_netcdf.pytopo_AG_AC_KX1r2_0.grd')
    # dem_B = import_dem('topo_AG_AC_KX1r2_0.grd')
    #
    # dem_diff = dem_A-dem_B
    #
    # fig, ax1 = plt.subplots(1, 1)
    #
    # dem_diff.to_array().plot(ax=ax1,vmin=-1000,vmax=1000)
    # # ax1.set_title('RMS of residuals (m):')
    #
    # fig.savefig(
    #     tmpdir + 'dem_diff.png',dpi=2000)
    #
    # exit()
