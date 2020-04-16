#!/usr/bin/env python3
# ----------------------------------
# Plot altimetry residuals (interpolated) over Mercury surface
# ----------------------------------
# Author: Stefano Bertone
# Created: 5-Nov-2019
#

import glob
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

import pickleIO
from eval_sol import draw_map
from prOpt import outdir, tmpdir, vecopts
from project_coord import project_stereographic

subdir = "" # archived/KX1r2_fitglborb/"
subexp = '3res_20amp'


def plot_topo(df):

    fig = plt.figure(figsize=(8, 6), edgecolor='w')
    # m = Basemap(projection='moll', resolution=None,
    #             lat_0=0, lon_0=0)
    m = Basemap(projection='npstere', boundinglat=10, lon_0=0, resolution='l')
    x, y = m(df.LON.values, df.LAT.values)
    map = m.scatter(x, y, c=df['R'].values, s=0.1,
                    cmap='Reds')  # afmhot') # , marker=',', s=3**piv.count(),
    plt.colorbar(map)
    draw_map(m)
    fig.savefig(tmpdir + 'mla_altres_' + sol + '_' + subexp + '.png')
    plt.clf()
    plt.close()

    deg_step = 1
    to_bin = lambda x: np.floor(x / deg_step) * deg_step
    df["latbin"] = df.LAT.map(to_bin)
    df["lonbin"] = df.LON.map(to_bin)
    groups = df.groupby(["latbin", "lonbin"])
    tmpdf = groups.R.apply(lambda x: np.median(x)).reset_index()
    # piv = pd.pivot_table(tmpdf, values="R", index=["latbin"], columns=["lonbin"], fill_value=0)
    # lats = np.deg2rad(piv.index.values) + np.pi / 2.
    # # TODO not sure why sometimes it has to be rescaled
    # lons = np.deg2rad(piv.columns.values) + np.pi
    # data = piv.values
    #
    print("Done groupby and medians")

    # Exclude last column because only 0<=lat<pi
    # and 0<=lon<pi are accepted (checked that lon=0 has same values)
    # print(data[:,0]==data[:,-1])
    # kx=ky=1 required not to mess up results!!!!!!!!!! Higher interp orders mess up...
    # interp_spline = RectBivariateSpline(lats[:-1],
    #                                     lons[:-1],
    #                                     data[:-1, :-1], kx=2, ky=2)
    # interp_spline = RectBivariateSpline(tmpdf.latbin.values,
    #                                     tmpdf.lonbin.values,
    #                                     tmpdf.R.values, kx=2, ky=2)
    start = time.time()

    import scipy.interpolate as interp
    x,y = project_stereographic(tmpdf.lonbin.values, tmpdf.latbin.values, 0, 90, R=vecopts['PLANETRADIUS'])

    def euclidean_norm_numpy(x1, x2):
        return np.linalg.norm(x1 - x2, axis=0)

    zfun_smooth_rbf = interp.Rbf(x, y, tmpdf.R.values, function='gaussian', norm=euclidean_norm_numpy,smooth=0)
    end = time.time()
    print('----- Runtime interp = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')

    # zfun_smooth_rbf = interp.Rbf(x, y, tmpdf.R.values, function='cubic',
    #                              smooth=0)  # default smooth=0 for interpolation
    pickleIO.save(zfun_smooth_rbf, tmpdir+"interp_R_"+sol+".pkl")
    # new_lats = np.deg2rad(np.arange(0, 180, 1))
    # new_lons = np.deg2rad(np.arange(0, 360, 1))
    start = time.time()

    xx, yy = np.mgrid[-2000:2000:100j, -2000:2000:100j]
    import dask.array as da

    n1 = xx.shape[1]
    ix = da.from_array(xx, chunks=(1, n1))
    iy = da.from_array(yy, chunks=(1, n1))
    iz = da.map_blocks(zfun_smooth_rbf, ix, iy)
    z_dense_smooth_rbf = iz.compute()
    # z_dense_smooth_rbf = zfun_smooth_rbf(new_lats,
    #                                      new_lons)  # not really a function, but a callable class instance
    fig, ax1 = plt.subplots(nrows=1)
    im = ax1.imshow(z_dense_smooth_rbf, origin='lower', cmap="RdBu")  # vmin=1,vmax=20,cmap="RdBu")
    fig.colorbar(im, ax=ax1, orientation='horizontal')
    fig.savefig(tmpdir + 'test_interp_' + sol + '.png')
    end = time.time()
    print('----- Runtime eval = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')

    # exit()
    #
    # z_dense_smooth_griddata = interp.griddata(np.array([tmpdf.latbin.values.ravel()+90.,tmpdf.lonbin.values.ravel()+180.]).T,
    #                                           tmpdf.R.values.ravel(),
    #                                           (new_lats, new_lons), method='cubic')
    # fig, ax1 = plt.subplots(nrows=1)
    # im = ax1.imshow(z_dense_smooth_griddata, origin='lower', cmap="RdBu")  # vmin=1,vmax=20,cmap="RdBu")
    # fig.colorbar(im, ax=ax1, orientation='horizontal')
    # fig.savefig(tmpdir + 'test_interp_' + sol + '.png')
    # exit()
    # pickleIO.save(interp_spline, tmpdir+"interp_R_"+sol+".pkl")
    # print("Done interp")
    #
    # new_lats = np.deg2rad(np.arange(0, 180, 1))
    # new_lons = np.deg2rad(np.arange(0, 360, 1))
    # new_lats, new_lons = np.meshgrid(new_lats, new_lons)
    # ev = interp_spline.ev(new_lats.ravel(), new_lons.ravel()).reshape((360, 180)).T
    # fig, ax1 = plt.subplots(nrows=1)
    # im = ax1.imshow(ev, origin='lower', cmap="RdBu")  # vmin=1,vmax=20,cmap="RdBu")
    # fig.colorbar(im, ax=ax1, orientation='horizontal')
    # fig.savefig(tmpdir + 'test_interp_' + sol + '.png')

    return interp_spline


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

    sols = ['tp4_0'] #,'KX1r2_17']
    topomaps = []

    new_lats = np.deg2rad(np.arange(0, 180, 1))
    new_lons = np.deg2rad(np.arange(0, 360, 1))
    new_lats, new_lons = np.meshgrid(new_lats, new_lons)

    for sol in sols:
        if True:
            files = glob.glob(outdir+"sim/"+subdir+sol+"/"+subexp+"/gtrack_*/gtrack_*.pkl")
            name = subdir.split('/')[-1]
            print(name)
            dflist = []
            for f in files:
                data = pd.read_pickle(f).ladata_df
                if len(data)>0:
                   dflist.append(data[['LON','LAT','R']])

            dflist = pd.concat(dflist)
            print("Done read+concat")

            dflist.to_csv(tmpdir+"ladata_concat_"+sol+".txt", sep='\t', index=False,header=False)
    exit()
