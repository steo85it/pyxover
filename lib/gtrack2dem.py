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

def plot_hillshade(data_grd):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cbook import get_sample_data
    from matplotlib.colors import LightSource

    from lib.gtrack2dem import import_dem
    from prOpt import tmpdir

    with import_dem(data_grd) as dem:  # np.load(get_sample_data('jacksboro_fault_dem.npz')) as dem: #

        print(dem.data[0].shape)
        xrange = [2000, 4000]
        yrange = [500, 2500]
        z = dem.data[0][xrange[0]:xrange[1], yrange[0]:yrange[1]]
        # z = dem['elevation']

        # -- Optional dx and dy for accurate vertical exaggeration ----------------
        # If you need topographically accurate vertical exaggeration, or you don't
        # want to guess at what *vert_exag* should be, you'll need to specify the
        # cellsize of the grid (i.e. the *dx* and *dy* parameters).  Otherwise, any
        # *vert_exag* value you specify will be relative to the grid spacing of
        # your input data (in other words, *dx* and *dy* default to 1.0, and
        # *vert_exag* is calculated relative to those parameters).  Similarly, *dx*
        # and *dy* are assumed to be in the same units as your input z-values.
        # Therefore, we'll need to convert the given dx and dy from decimal degrees
        # to meters.
        dx, dy = 1, 1  # dem['dx'], dem['dy']

        dy = 111200 * dy
        dx = 111200 * dx * np.cos(np.radians(-1500.))  # dem['ymin']))
        # -------------------------------------------------------------------------

    # choose vert_exag
    vert_exag = [100, 1000, 10000]

    # Shade from the northwest, with the sun 45 degrees from horizontal
    ls = LightSource(azdeg=315, altdeg=5)
    cmap = plt.cm.gist_earth

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 9))
    plt.setp(axes.flat, xticks=[], yticks=[])

    # Vary vertical exaggeration and blend mode and plot all combinations
    for col, ve in zip(axes.T, vert_exag):
        # Show the hillshade intensity image in the first row
        col[0].imshow(ls.hillshade(z, vert_exag=ve, dx=dx, dy=dy), cmap='gray')

        # Place hillshaded plots with different blend modes in the rest of the rows
        for ax, mode in zip(col[1:], ['hsv', 'overlay', 'soft']):
            rgb = ls.shade(z, cmap=cmap, blend_mode=mode,
                           vert_exag=ve, dx=dx, dy=dy)
            ax.imshow(rgb)

    # Label rows and columns
    for ax, ve in zip(axes[0], vert_exag):
        ax.set_title('{0}'.format(ve), size=18)
    for ax, mode in zip(axes[:, 0], ['Hillshade', 'hsv', 'overlay', 'soft']):
        ax.set_ylabel(mode, size=18)

    # Group labels...
    axes[0, 1].annotate('Vertical Exaggeration', (0.5, 1), xytext=(0, 30),
                        textcoords='offset points', xycoords='axes fraction',
                        ha='center', va='bottom', size=20)
    axes[2, 0].annotate('Blend Mode', (0, 0.5), xytext=(-30, 0),
                        textcoords='offset points', xycoords='axes fraction',
                        ha='right', va='center', size=20, rotation=90)
    fig.subplots_adjust(bottom=0.05, right=0.95)

    plt.savefig(tmpdir + "test_hillshade.png")

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

    sols = ['BS2_0'] #AGTP_0'] # tp4_0'] #,'KX1r2_17']
    subexp = '0res_1amp'
    # topomaps = []

    concat_data = False # True
    create_grid = True # True
    create_hillshade = True

    for sol in sols:
        csvfile = "ladata_concat_" + sol + ".txt"
        topofile = "topo_" + sol + ".grd"

        # import geolocalised elevations to single csv
        # ============================================
        if concat_data:
            files = glob.glob(outdir + "sim/" + sol + "/" + subexp + "/gtrack_*/gtrack_*.pkl")
            concat_elevations(files)
            print("Done read+concat")
        else:
            print("Concat = False")

        # generate grd map
        # ================
        if create_grid:
            # print('mapproject', csvfile,'-Js0/90/1:1000 -R0/360/75/90 -S -C -bo | blockmedian -bi3 -C -r -I.05 -R-630/630/-630/630 -bo3 | surface -bi3 -T0.75 -r -I.05 -R-630/630/-630/630 -Gtopo_tp4_0.grd')
            command = ["gmt mapproject "+csvfile+" -Js0/90/1:100000 -R0/360/60/90 -S -C -bo | "
                                             "gmt blockmedian -bi3 -C -r -I.1 -R-1500/1500/-1500/1500 -bo3 | "
                                             "gmt surface -bi3 -T0.25 -r -I.1 -R-1500/1500/-1500/1500 -G"+topofile]
            r_dem = subprocess.check_output(command, universal_newlines=True, shell=True, cwd=tmpdir)
            print("Done generate grd")
        else:
            print("create_grid = False")

        # import and plot DEM interpolated from residuals
        # ===============================================
        if create_hillshade:

            plot_hillshade(data_grd=tmpdir+topofile)

            # dem_xarray = import_dem(tmpdir+topofile)
            # print(dem_xarray)
            # print(dem_xarray.data[0].shape)
            # exit()
            #
            # xrange = [700,1000]
            # yrange = [1500,1900]
            # data = dem_xarray.data[0][xrange[0]:xrange[1],yrange[0]:yrange[1]]
            #
            # plot_multiple(data)
            print("Done plotting topo")
        else:
            print("Create topo = False")


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
