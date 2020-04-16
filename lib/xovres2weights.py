#!/usr/bin/env python3
# ----------------------------------
# Determine data weights from residuals map or roughness map
# ----------------------------------
# Author: Stefano Bertone
# Created: 6-Sep-2019
#

import pickle

import seaborn as sns
from prOpt import tmpdir, auxdir, local, debug, outdir, vecopts, roughn_map
from util import rms

import pyproj
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline #, interpolate, interp2d

import pickleIO
from xov_setup import xov
import pandas as pd

# use roughness map from Kreslavski et al, GRL, 2014
# (https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2014GL062162)
# to associate error to each xover based on separation from MLA observations
def get_interpolation_weight(xov_):

    interp_weights = xov_.xovers.copy()
    interp_weights = interp_weights[['LON','LAT','dist_min_mean']]

    # get roughness at 700 meters from http://www.planetary.brown.edu/html_pages/mercury_roughness-maps.html
    # ref baseline in meters (depending on input map)
    ref_baseline = 700.
    lonlat = interp_weights.loc[:,['LON','LAT']].values
    roughness_df = get_roughness_at_coord(lon=lonlat[:,0],lat=lonlat[:,1],
                                          roughness_map=auxdir+'MLA_Roughness_composite.tif')
    interp_weights['rough_700m'] = roughness_df.loc[:,'rough_700m']

    # get min separation between xovers and mla data (meters, mean of minimal sep for each track ... guess it makes sense)
    interp_weights['meters_dist_min'] = interp_weights.dist_min_mean.values*1.e3 # filter(regex='dist_[A,B].*').min(axis=1).values*1.e3

    # get roughness at separation baseline
    interp_weights['rough_at_mindist'] = roughness_at_baseline(interp_weights['rough_700m'].values,
                                                            interp_weights['meters_dist_min'].values,
                                                               ref_baseline=ref_baseline)

    # get weight as inverse of roughness (relative, 0:1) value - could use factors given in ref + sim results to rescale
    interp_weights['weight'] = 1./interp_weights['rough_at_mindist'].values

    print(interp_weights)
    print(interp_weights['weight'].mean(),interp_weights['weight'].median(),interp_weights['weight'].max(),interp_weights['weight'].min())

    # plot some histos for debug
    if debug:
        plt.figure() #figsize=(8, 3))
        num_bins = 'auto' # 40  # v

        tmp = interp_weights['rough_700m'].values
        n, bins, patches = plt.hist(tmp.astype(np.float), bins=num_bins, cumulative=True)
        plt.xlabel('roughness@baseline700 (m/m)')
        plt.savefig(tmpdir + '/histo_roughn_at_700.png')
        plt.clf()

        tmp = interp_weights['meters_dist_min'].values
        n, bins, patches = plt.hist(np.where(tmp < 500., tmp, 500.).astype(np.float), bins=num_bins, cumulative=True)
        plt.xlabel('distance (m)')
        plt.savefig(tmpdir + '/histo_interp_dist0.png')
        plt.clf()

        tmp = interp_weights['rough_at_mindist'].values
        n, bins, patches = plt.hist(np.where(tmp<255,tmp,255).astype(np.float), bins=num_bins, cumulative=True)
        plt.xlabel('roughness@separation (m/m)')
        plt.xlim([0,10])
        plt.savefig(tmpdir + '/histo_roughn_at_sep.png')
        plt.clf()

        tmp = interp_weights['weight'].values
        n, bins, patches = plt.hist(tmp.astype(np.float), bins=num_bins, cumulative=False)
        plt.xlabel('weight')
        plt.savefig(tmpdir + '/histo_roughn_weight.png')
        plt.clf()

        tmp = interp_weights['weight'].values
        n, bins, patches = plt.hist(tmp.astype(np.float), bins=num_bins, cumulative=True)
        plt.xlabel('roughness@separation (m/m)')
        plt.savefig(tmpdir + '/histo_roughn_weights.png')
        plt.clf()

    return interp_weights


# old function to get roughness from residuals, then use interpolated roughness and literature to
# extrapolate roughness at separation=baseline, then reconvert to expected rms
def get_weight_regrough(xov_, tstnam='', new_map=False):

    regbas_weights = xov_.xovers.copy()

    # TODO This screens data and modifies xov!!!!
    if new_map:
        generate_dR_regions(filnam=tstnam,xov=regbas_weights)
        # exit()

    new_lats = np.deg2rad(np.arange(0, 180, 1))
    new_lons = np.deg2rad(np.arange(0, 360, 1))
    new_lats, new_lons = np.meshgrid(new_lats, new_lons)

    if tstnam != '':
        interp_spline = pickleIO.load(auxdir+"interp_dem_"+tstnam+".pkl") #/interp_dem.pkl")
    else:
        interp_spline = pickleIO.load(auxdir+"interp_dem_tp8_0.pkl") #/interp_dem.pkl")

    ev = interp_spline.ev(new_lats.ravel(),new_lons.ravel()).reshape((360, 180)).T

    # compare maps
    if False:
        interp_spline = pickleIO.load(auxdir + "interp_dem_KX1r2_10.pkl")  # tp8_0.pkl") #/interp_dem.pkl")
        ev1 = interp_spline.ev(new_lats.ravel(), new_lons.ravel()).reshape((360, 180)).T
        ev = ev-ev1

    # print(ev)

    if local and debug:
        fig, ax1 = plt.subplots(nrows=1)
        im = ax1.imshow(ev,origin='lower',cmap="RdBu") # vmin=1,vmax=20,cmap="RdBu")
        fig.colorbar(im, ax=ax1,orientation='horizontal')
        fig.savefig(tmpdir+'test_interp_'+tstnam+'.png')

    regbas_weights['region'] = get_demz_at(interp_spline,regbas_weights['LAT'].values,regbas_weights['LON'].values)
    step = 10
    regbas_weights['region'] = np.floor(regbas_weights['region'].values / step) * step
    regbas_weights['reg_rough_150'] = regrms_to_regrough(regbas_weights['region'].values)
    regbas_weights['meters_dist_min'] = regbas_weights.filter(regex='dist_[A,B].*').min(axis=1).values
    regbas_weights['rough_at_mindist'] = roughness_at_baseline(regbas_weights['reg_rough_150'].values,
                                                            regbas_weights['meters_dist_min'].values)
    # xov_.xovers['rough_at_max'] = roughness_at_baseline(xov_.xovers['reg_rough_150'].values,
    #                                                         xov_.xovers.filter(regex='dist_max').values)

    regbas_weights = regbas_weights[['region','reg_rough_150','meters_dist_min','rough_at_mindist','dR']]
                       # 'rough_at_max','dR']])
    regbas_weights['error'] = roughness_to_error(regbas_weights.loc[:,'rough_at_mindist'].values)

    if debug:
        print(regbas_weights)
        print(regbas_weights[['region','reg_rough_150','meters_dist_min','rough_at_mindist','dR']].corr()) #,
                           # 'rough_at_max','dR']].corr())

    # exit()

    # fig = plt.figure(figsize=(10, 8), edgecolor='w')
    # m = Basemap(projection='moll', #'cea', #
    #             resolution=None,
    #             lat_0=0, lon_0=180)
    #
    # # m = Basemap(projection='merc',llcrnrlat=-88,urcrnrlat=88,\
    # #             llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
    #
    # x, y = m(mladR.lonbin.values, mladR.latbin.values)
    # m.scatter(x, y, marker=',',c=mladR.dR,cmap='Reds') # , s=mladR.dR
    # draw_map(m)
    #
    # fig.savefig(tmpdir + 'mla_count_moll_KX0.png')
    # plt.clf()
    # plt.close()
    #
    # exit()
    return regbas_weights


def generate_dR_regions(filnam,xov):
    # pass
    #xov_ = xov.copy()
    # print(xov_.xov)
    # exit()
    xovtmp = xov.copy()

    if len(xovtmp)>0:
        xovtmp['dist_max'] = xovtmp.filter(regex='^dist_.*$').max(axis=1)
        xovtmp['dist_minA'] = xovtmp.filter(regex='^dist_A.*$').min(axis=1)
        xovtmp['dist_minB'] = xovtmp.filter(regex='^dist_B.*$').min(axis=1)
        xovtmp['dist_min_avg'] = xovtmp.filter(regex='^dist_min.*$').mean(axis=1)

        print("pre weighting")
        print(xovtmp.dR.abs().max(),xovtmp.dR.abs().min(),xovtmp.dR.median(),rms(xovtmp.dR.values))
        print("post weighting")
        xovtmp.dR *= xovtmp.huber
        print(xovtmp.dR.abs().max(),xovtmp.dR.abs().min(),xovtmp.dR.median(),rms(xovtmp.dR.values))
        #
        # remove data if xover distance from measurements larger than 0.4km (interpolation error)
        # plus remove outliers with median method
        # xovtmp = xovtmp[xovtmp.dist_max < 0.4]
        # print(len(xovtmp[xovtmp.dist_max > 0.4]),
        #       'xovers removed by dist_max from obs > 0.4km')
        # if sim_altdata == 0:
        #mean_dR, std_dR, worse_tracks = xov_.remove_outliers('dR',remove_bad=True)

    # dR absolute value taken
    xovtmp['dR_orig'] = xovtmp.dR
    xovtmp.dR *= xovtmp.huber
    # print(xovtmp['dR'])
    # print(rms(xovtmp['dR'].values),xovtmp.dR.abs().median())

    deg_step = 5
    to_bin = lambda x: np.floor(x / deg_step) * deg_step
    xovtmp["latbin"] = xovtmp.LAT.map(to_bin)
    xovtmp["lonbin"] = xovtmp.LON.map(to_bin)
    groups = xovtmp.groupby(["latbin", "lonbin"])
    # print(groups.size().reset_index())

    xov_.save(tmpdir +filnam+"_clean_grp.pkl")

    mladR = groups.dR.apply(lambda x: rms(x)).reset_index() #median().reset_index() #
    mladR['count'] = groups.size().reset_index()[[0]]
    mladR.loc[mladR['count'] < 1, 'dR'] = rms(xovtmp['dR'].values)
    # print(mladR)
    # mladR = xovtmp.round({'LON': 0, 'LAT': 0, 'dR': 3}).groupby(['LON', 'LAT']).dR.median().reset_index()
    # print(mladR)
    dRstep = 5
    mladR['dR'] = np.floor(mladR['dR'].values / dRstep) * dRstep
    # print(mladR.sort_values(by='count'))
    # exit()
    global_rms = np.floor(rms(xovtmp['dR'].values) / dRstep) * dRstep

    fig, ax1 = plt.subplots(nrows=1)
    # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
    # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
    # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
    # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    piv = pd.pivot_table(mladR, values="dR", index=["latbin"], columns=["lonbin"], fill_value=global_rms)
    # plot pivot table as heatmap using seaborn
    sns.heatmap(piv, xticklabels=10, yticklabels=10,cmap="RdBu") #,vmin=14,vmax=17,cmap="RdBu")
    plt.tight_layout()
    ax1.invert_yaxis()
    #         ylabel='Topog ampl rms (1st octave, m)')
    fig.savefig(tmpdir + '/mla_dR_'+filnam+'.png')
    plt.clf()
    plt.close()

    lats = np.deg2rad(piv.index.values) + np.pi / 2.
    # TODO not sure why sometimes it has to be rescaled
    lons = np.deg2rad(piv.columns.values) + np.pi
    data = piv.values

    print(data)

    # Exclude last column because only 0<=lat<pi
    # and 0<=lon<pi are accepted (checked that lon=0 has same values)
    # print(data[:,0]==data[:,-1])
    # kx=ky=1 required not to mess up results!!!!!!!!!! Higher interp orders mess up...
    interp_spline = RectBivariateSpline(lats[:-1],
                                        lons[:-1],
                                        data[:-1, :-1], kx=1, ky=1)
    pickleIO.save(interp_spline, auxdir+"interp_dem_"+filnam+".pkl")

    return 0

def get_roughness_at_coord(lon,lat,roughness_map):
    """
    Interpolate input map to get (relative) roughness - for the template map, 0 = smooth, 255 = rough
    :param lon:
    :param lat:
    :param roughness_map:
    :return:
    """
    da = xr.open_rasterio(roughness_map)

    if False and debug and local:
        fig = plt.figure(figsize=(12, 8))
        da.plot.imshow()
        plt.savefig(tmpdir + 'test_roughness.png')

    # select 0.7 km baseline only
    da = da.loc[dict(band=3)]

    # ###########################
    # rough500_map = pd.read_csv('/home/sberton2/Downloads/0.50km_g_for_gmt.xyz',header=None,names=['lon','lat','rough_500m'])
    # print(rough500_map)
    # xds = xr.Dataset.from_dataframe(rough500_map)
    # print(xds)
    # exit()
    #
    # ###########################

    # reproject xovers coords in roughness dataArray (geotif) projection
    p = pyproj.Proj(da.attrs['crs'])
    x, y = p(lon,lat)

    # reindex array to get interpolated roughness(x,y) only, and not on full mesh
    x = xr.DataArray(x, dims='z')
    y = xr.DataArray(y, dims='z')
    dai = da.interp(x=x, y=y)
    roughness_df = pd.DataFrame([x.values, y.values, dai.data]).T.fillna(0)
    roughness_df['LON'] = lon
    roughness_df['LAT'] = lat
    roughness_df.columns = ['x_laea', 'y_laea', 'rough_700m', 'LON', 'LAT']

    # set standard roughness for rest of planet ()
    if roughn_map:
        roughness_df['rough_700m'] /= 255.
        roughness_df.loc[(roughness_df['LAT'] < 65) | (roughness_df['LAT'] > 84), 'rough_700m'] = 0.1
        roughness_df.loc[roughness_df['rough_700m']<1.e-2] = 1.e-2
    else:
    # if simulation with fixed small scale of 20 mt at 600 mt (important thing is that it's the same everywhere)
        print("ACHTUNG!!!! Roughness fixed at constant value for simulations!!!!!")
        roughness_df.loc[:,'rough_700m'] = 0.1
    # roughness = roughness.round({'LAT':0,'LON':0, 'rough_700m':0})

    # check if interp and axes are aligned
    if debug and local:
        # in lambert azimutal equal area projection
        fig = plt.figure() #figsize=(12, 8))
        plt.xlim((-1.e6, 1.e6))
        plt.ylim((-1.e6, 1.e6))
        # plt.tricontour(roughness['x_laea'].values, roughness['y_laea'].values, roughness['rough_700m'].values, 15, linewidths=0.5, colors='k')
        plt.tricontourf(roughness_df['x_laea'].values, roughness_df['y_laea'].values, roughness_df['rough_700m'].values, 15)
        plt.savefig(tmpdir + 'test_roughness_interp2.png')
        # in lat lon to see if frames are aligned
        plt.clf()
        fig = plt.figure() #figsize=(12, 8))
        plt.ylim((65,84))
        # plt.tricontour(roughness['x_laea'].values, roughness['y_laea'].values, roughness['rough_700m'].values, 15, linewidths=0.5, colors='k')
        plt.tricontourf(roughness_df['LON'].values, roughness_df['LAT'].values, roughness_df['rough_700m'].values, 15)
        plt.savefig(tmpdir + 'test_roughness_interp3.png')

    return roughness_df

# extrapolate roughness at baseline (meters) from given roughness at reference baseline (meters)
def roughness_at_baseline(roughness_at_ref_baseline, out_baseline, ref_baseline=150):
    """
    :param roughness_at_ref_baseline: input roughness at reference baseline
    :param out_baseline: output baseline (e.g., separation btw xovers and MLA data) in meters
    :param ref_baseline: reference baseline in meters
    :return: roughness at out_baseline
    """

    # 0.65 = persistence factor of fractal noise used in simu (approx ln(2))
    result = roughness_at_ref_baseline / np.power((2 * 0.65), np.log2(ref_baseline / out_baseline))

    return result

def regrms_to_regrough(rms):

    roughness = (rms - 7.39) / 0.64
    # fix minimum roughness to 4 meters @ 150 meters (similar to Moon, 0 does not make sense)
    return np.where(roughness>4, roughness, 4.)

def roughness_to_error(roughness):

    return (0.64 * roughness) + 7.39

def get_demz_at(dem_xarr, lattmp, lontmp):
    # lontmp += 180.
    lontmp[lontmp < 0] += 360.
    # print("eval")
    # print(np.sort(lattmp))
    # print(np.sort(lontmp))
    # print(np.sort(np.deg2rad(lontmp)))
    # exit()

    return dem_xarr.ev(np.deg2rad(lattmp)+np.pi/2., np.deg2rad(lontmp))

if __name__ == '__main__':

    test = 'KX1r2' # 'tp8' #
    topo = '0res_1amp'

    if False:
        for i in [0,1,5,10]: #range(14):
            filnam = outdir+'/sim/'+test+'_'+str(i)+'/'+topo+'/Abmat_sim_'+test+'_'+str(i+1)+'_'+topo+'.pkl'
            print(filnam)
            vecopts = {}
            xov_ = xov(vecopts)
            xov_ = xov_.load(filnam)
            print(xov_.__dict__)
            #xov_ = xov_.xov #tmpdir+"Abmat_sim_"+filnam+"_0res_1amp.pkl")
            print("Loaded...")
            print(xov_.xov.xovers)
            testname = test+'_'+str(i)
            run(xov_.xov,testname,new_map=True)

    i = 0
    filnam = outdir + '/sim/' + test + '_' + str(i) + '/' + topo + '/Abmat_sim_' + test + '_' + str(
        i + 1) + '_' + topo + '.pkl'
    amat_ = xov(vecopts)
    amat_ = amat_.load(filnam)

    interperr_weights_df = get_interpolation_weight(amat_.xov)

    # lonlat = xov_.xov.xovers.loc[:,['LON','LAT']].values
    # 
    # get_roughness_at_coord(lon=lonlat[:,0],lat=lonlat[:,1],roughness_map='/home/sberton2/Downloads/MLA_Roughness_composite.tif')
