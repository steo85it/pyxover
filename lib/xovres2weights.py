#!/usr/bin/env python3
# ----------------------------------
# Determine data weights from residuals map
# ----------------------------------
# Author: Stefano Bertone
# Created: 6-Sep-2019
#

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from mpl_toolkits.basemap import Basemap
from scipy.interpolate import RectBivariateSpline

import pickleIO
#from eval_sol import rmse#, draw_map
from prOpt import tmpdir, auxdir, local, debug
from util import rms
from xov_setup import xov

def run(xov_,tstnam=''):

    regbas_weights = xov_.xovers.copy()

    # TODO This screens data and modifies xov!!!!
    if False:
        generate_dR_regions(filnam=tstnam,xov=xov_)
        exit()

    new_lats = np.deg2rad(np.arange(0, 180, 1))
    new_lons = np.deg2rad(np.arange(0, 360, 1))
    new_lats, new_lons = np.meshgrid(new_lats, new_lons)

    interp_spline = pickleIO.load(tmpdir+"interp_dem_KX1r_0.pkl") #/interp_dem.pkl")

    ev = interp_spline.ev(new_lats.ravel(),new_lons.ravel()).reshape((360, 180)).T

    # print(ev)

    if local:
        fig, ax1 = plt.subplots(nrows=1)
        im = ax1.imshow(ev,origin='lower',vmin=10,vmax=50,cmap="RdBu")
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
    xov_ = xov.copy()
    # print(xov_.xov)
    # exit()
    xovtmp = xov_.xovers.copy()

    if len(xovtmp)>0:
        xovtmp['dist_max'] = xovtmp.filter(regex='^dist_.*$').max(axis=1)
        xovtmp['dist_minA'] = xovtmp.filter(regex='^dist_A.*$').min(axis=1)
        xovtmp['dist_minB'] = xovtmp.filter(regex='^dist_B.*$').min(axis=1)
        xovtmp['dist_min_avg'] = xovtmp.filter(regex='^dist_min.*$').mean(axis=1)

        # remove data if xover distance from measurements larger than 5km (interpolation error)
        # plus remove outliers with median method
        xovtmp = xovtmp[xovtmp.dist_max < 0.4]
        print(len(xovtmp[xovtmp.dist_max > 0.4]),
              'xovers removed by dist_max from obs > 0.4km')
        # if sim_altdata == 0:
        mean_dR, std_dR, worse_tracks = xov_.remove_outliers('dR',remove_bad=True)

    # dR absolute value taken
    xovtmp['dR_orig'] = xovtmp.dR
    xovtmp['dR'] = xovtmp.dR.abs()

    step = 3
    to_bin = lambda x: np.floor(x / step) * step
    xovtmp["latbin"] = xovtmp.LAT.map(to_bin)
    xovtmp["lonbin"] = xovtmp.LON.map(to_bin)
    groups = xovtmp.groupby(("latbin", "lonbin"))

    xov_.save(tmpdir +filnam+"_clean_grp.pkl")

    mladR = groups.dR.apply(lambda x: rms(x)).reset_index() #median().reset_index() #
    # exit()
    # mladR = xovtmp.round({'LON': 0, 'LAT': 0, 'dR': 3}).groupby(['LON', 'LAT']).dR.median().reset_index()
    # print(mladR)
    step = 0.1
    mladR['dR'] = np.floor(mladR['dR'].values / step) * step
    # print(mladR)

    # exit()


    fig, ax1 = plt.subplots(nrows=1)
    # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
    # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
    # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
    # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    piv = pd.pivot_table(mladR, values="dR", index=["latbin"], columns=["lonbin"], fill_value=0)
    # plot pivot table as heatmap using seaborn
    piv = piv.fillna(0)  # (piv + empty_geomap_df).fillna(0)
    # print(piv)
    # exit()
    # cmap = sns.set_palette(sns.color_palette("RdBu", 10))
    # sns.heatmap(piv, xticklabels=10, yticklabels=10,vmin=10,vmax=40,center=30,cmap="RdBu")
    sns.heatmap(piv, xticklabels=10, yticklabels=10,vmin=0,vmax=30,cmap="RdBu")
    plt.tight_layout()
    ax1.invert_yaxis()
    #         ylabel='Topog ampl rms (1st octave, m)')
    fig.savefig(tmpdir + '/mla_dR_'+filnam+'.png')
    plt.clf()
    plt.close()

    lats = np.deg2rad(piv.index.values) + np.pi / 2.
    lons = np.deg2rad(piv.columns.values)  # -np.pi
    data = piv.values

    print(data)

    # Exclude last column because only 0<=lat<pi
    # and 0<=lon<pi are accepted (checked that lon=0 has same values)
    # print(data[:,0]==data[:,-1])
    # kx=ky=1 required not to mess up results!!!!!!!!!! Higher interp orders mess up...
    interp_spline = RectBivariateSpline(lats[:-1],
                                        lons[:-1],
                                        data[:-1, :-1], kx=1, ky=1)
    pickleIO.save(interp_spline, tmpdir+"interp_dem_"+filnam+".pkl")

    return 0


def roughness_at_baseline(regional_roughness,baseline):

    baseline *= 1.e3
    # 0.65 = persistence factor of fractal noise used in simu (approx ln(2))
    return regional_roughness/ np.power((2*0.65),np.log2(150./baseline))

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

    test = 'KX1r'
    topo = '0res_1amp'

    for i in range(9):
        filnam = test+'_'+str(i)
        vecopts = {}
        xov_ = xov(vecopts)
        xov_ = xov_.load('/home/sberton2/Works/NASA/Mercury_tides/out/sim/'+test+'_'+str(i)+'/'+topo+'/Abmat_sim_'+test+'_'+str(i+1)+'_'+topo+'.pkl')
        xov_ = xov_.xov #tmpdir+"Abmat_sim_"+filnam+"_0res_1amp.pkl")
        print("Loaded...")
        print(xov_)

        run(xov_,filnam)
