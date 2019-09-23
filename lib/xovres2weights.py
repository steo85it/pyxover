#!/usr/bin/env python3
# ----------------------------------
# Determine data weights from residuals map
# ----------------------------------
# Author: Stefano Bertone
# Created: 6-Sep-2019
#

import matplotlib.pyplot as plt
import numpy as np

import pickleIO
from prOpt import tmpdir, auxdir


def run(xov_):
    # vecopts = {}
    # xov_ = xov(vecopts)
    # xov_ = xov_.load(tmpdir+"Abmat_sim_KX1_1_0res_1amp.pkl").xov
    # print("Loaded...")
    # print(xov_)
    #
    # if len(xov_.xovers)>0:
    #     xov_.xovers['dist_max'] = xov_.xovers.filter(regex='^dist_.*$').max(axis=1)
    #     xov_.xovers['dist_minA'] = xov_.xovers.filter(regex='^dist_A.*$').min(axis=1)
    #     xov_.xovers['dist_minB'] = xov_.xovers.filter(regex='^dist_B.*$').min(axis=1)
    #     xov_.xovers['dist_min_avg'] = xov_.xovers.filter(regex='^dist_min.*$').mean(axis=1)
    #
    #     # remove data if xover distance from measurements larger than 5km (interpolation error)
    #     # plus remove outliers with median method
    #     xov_.xovers = xov_.xovers[xov_.xovers.dist_max < 0.4]
    #     print(len(xov_.xovers[xov_.xovers.dist_max > 0.4]),
    #           'xovers removed by dist_max from obs > 0.4km')
    #     # if sim_altdata == 0:
    #     mean_dR, std_dR, worse_tracks = xov_.remove_outliers('dR',remove_bad=True)
    #
    # # dR absolute value taken
    # xov_.xovers['dR_orig'] = xov_.xovers.dR
    # xov_.xovers['dR'] = xov_.xovers.dR.abs()
    #
    # step = 3
    # to_bin = lambda x: np.floor(x / step) * step
    # xov_.xovers["latbin"] = xov_.xovers.LAT.map(to_bin)
    # xov_.xovers["lonbin"] = xov_.xovers.LON.map(to_bin)
    # groups = xov_.xovers.groupby(("latbin", "lonbin"))
    #
    # xov_.save(tmpdir + "KX1_clean_grp.pkl")
    #
    # mladR = groups.dR.apply(lambda x: rmse(x)).reset_index() #median().reset_index() #
    # # exit()
    # # mladR = xov_.xovers.round({'LON': 0, 'LAT': 0, 'dR': 3}).groupby(['LON', 'LAT']).dR.median().reset_index()
    # # print(mladR)
    # step = 10
    # mladR['dR'] = np.floor(mladR['dR'].values / step) * step
    # # print(mladR)
    #
    # # exit()
    #
    #
    # fig, ax1 = plt.subplots(nrows=1)
    # # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
    # # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
    # # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
    # # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
    # # Draw the heatmap with the mask and correct aspect ratio
    # piv = pd.pivot_table(mladR, values="dR", index=["latbin"], columns=["lonbin"], fill_value=0)
    # # plot pivot table as heatmap using seaborn
    # piv = piv.fillna(0)  # (piv + empty_geomap_df).fillna(0)
    # # print(piv)
    # # exit()
    # # cmap = sns.set_palette(sns.color_palette("RdBu", 10))
    # # sns.heatmap(piv, xticklabels=10, yticklabels=10,vmin=10,vmax=40,center=30,cmap="RdBu")
    # sns.heatmap(piv, xticklabels=10, yticklabels=10,vmin=20,vmax=60,center=40,cmap="RdBu")
    # plt.tight_layout()
    # ax1.invert_yaxis()
    # #         ylabel='Topog ampl rms (1st octave, m)')
    # fig.savefig(tmpdir + '/mla_dR_KX1_0.png')
    # plt.clf()
    # plt.close()
    #
    # lats = np.deg2rad(piv.index.values) + np.pi / 2.
    # lons = np.deg2rad(piv.columns.values)  # -np.pi
    # data = piv.values
    #
    # print(data)
    #
    # # Exclude last column because only 0<=lat<pi
    # # and 0<=lon<pi are accepted (checked that lon=0 has same values)
    # # print(data[:,0]==data[:,-1])
    # # kx=ky=1 required not to mess up results!!!!!!!!!! Higher interp orders mess up...
    # interp_spline = RectBivariateSpline(lats[:-1],
    #                                     lons[:-1],
    #                                     data[:-1, :-1], kx=1, ky=1)
    # pickleIO.save(interp_spline, "tmp/interp_dem_KX1.pkl")

    new_lats = np.deg2rad(np.arange(0, 180, 1))
    new_lons = np.deg2rad(np.arange(-180, 180, 1))
    new_lats, new_lons = np.meshgrid(new_lats, new_lons)

    interp_spline = pickleIO.load(tmpdir+"/interp_dem.pkl")

    ev = interp_spline.ev(new_lats.ravel(),new_lons.ravel()).reshape((360, 180)).T

    print(ev)

    fig, ax1 = plt.subplots(nrows=1)
    ax1.imshow(ev,origin='lower',vmin=20,vmax=60,cmap="RdBu")
    fig.savefig(auxdir+'test_interp_KX1.png')

    xov_.xovers['region'] = get_demz_at(interp_spline,xov_.xovers['LAT'].values,xov_.xovers['LON'].values)
    step = 10
    xov_.xovers['region'] = np.floor(xov_.xovers['region'].values / step) * step
    xov_.xovers['reg_rough_150'] = regrms_to_regrough(xov_.xovers['region'].values)
    xov_.xovers['dist_min'] = xov_.xovers.filter(regex='dist_[A,B].*').min(axis=1).values
    xov_.xovers['rough_at_mindist'] = roughness_at_baseline(xov_.xovers['reg_rough_150'].values,
                                                            xov_.xovers['dist_min'].values)
    # xov_.xovers['rough_at_max'] = roughness_at_baseline(xov_.xovers['reg_rough_150'].values,
    #                                                         xov_.xovers.filter(regex='dist_max').values)

    regbas_weights = xov_.xovers[['region','reg_rough_150','dist_min','rough_at_mindist','dR']] #,
                       # 'rough_at_max','dR']])
    print(regbas_weights)
    print(xov_.xovers[['region','reg_rough_150','dist_min','rough_at_mindist','dR']].corr()) #,
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

def roughness_at_baseline(regional_roughness,baseline):

    baseline *= 1.e3
    # 0.65 = persistence factor of fractal noise used in simu (approx ln(2))
    return regional_roughness/ np.power((2*0.65),np.log2(150./baseline))

def regrms_to_regrough(rms):

    return (rms - 7.39) / 0.64

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
    run()