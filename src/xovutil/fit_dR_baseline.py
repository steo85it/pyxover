#!/usr/bin/env python3
# ----------------------------------
# collect sim tests and fit rms
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#

# create data
import numpy as np
from scipy.stats import stats
import matplotlib.pyplot as plt

from src.accumxov import accum_opt, AccumXov as xovacc, accum_utils
from src.accumxov.Amat import Amat
# from examples.MLA.options import XovOpt.get("tmpdir"), XovOpt.get("outdir")
from config import XovOpt

from src.xovutil.stat import rms


def fit_sols(sol,exp_list):

    vecopts = {}
    list = []

    for subexp in exp_list:
        tmp = Amat(vecopts)
        tmp = tmp.load(XovOpt.get("outdir") + 'sim/' + sol + '/' + subexp + '/Abmat_sim_' + sol.split('_')[0] + '_' + str(int(sol.split('_')[-1]) + 1) + '_' + subexp + '.pkl')

        if tmp.xov.xovers.filter(regex='^dist_.*$').empty==False:

            tmp.xov.xovers['dist_max'] = tmp.xov.xovers.filter(regex='^dist_.*$').max(axis=1)
            tmp.xov.xovers['dist_minA'] = tmp.xov.xovers.filter(regex='^dist_A.*$').min(axis=1)
            tmp.xov.xovers['dist_minB'] = tmp.xov.xovers.filter(regex='^dist_B.*$').min(axis=1)
            tmp.xov.xovers['dist_min_mean'] = tmp.xov.xovers.filter(regex='^dist_min.*$').mean(axis=1)
            accum_utils.analyze_dist_vs_dR(tmp.xov)

            if accum_opt.remove_max_dist:
                print(len(tmp.xov.xovers[tmp.xov.xovers.dist_max > 0.4]),
                      'xovers removed by dist from obs > 400m')
                tmp.xov.xovers = tmp.xov.xovers[tmp.xov.xovers.dist_max < 0.4]
                tmp.xov.xovers = tmp.xov.xovers[tmp.xov.xovers.dist_min_mean < 1]

        # Remove huge outliers
        mean_dR, std_dR, worst_tracks = tmp.xov.remove_outliers('dR', remove_bad=accum_opt.remove_3sigma_median)

        list.append([subexp,mean_dR,std_dR,rms(tmp.xov.xovers.dR.values)])

    sols = np.array(list)
    y = np.array(list)[:,3].astype(float)
    print(y)

    x = np.array([2.5,5,10,15,20,30,40]) #,16,23])
    # if y = sim with tides and no errors -> horiz line,
    # else if tides not included in model, and no other errors, 1:1 bisec
    # y = np.array([4.8,6.3,9.8,13.4,17.2,24.6,32.2]) #,11.6,16])

    # linear regression
    # res = stats.theilslopes(y, x, 0.90)

    deg2fit = np.polyfit(x[:-2], y[:-2], 2)
    # deg3fit = np.polyfit(x[:-2], y[:-2], 3)

    print(x,y)

    res = stats.linregress(x,y) #[:-2], y[:-2])

    print('y={:.2f}x+{:.2f} +- std={:.2f}'.format(res[0],res[1],res[-1]))

    fig, ax1 = plt.subplots(1, 1)
    #ax1.plot(x,y,'k.')

    # print(y[:-2], res[1] + res[0] * x[:-2], x[:-2])
    # print("data tot: ",(y-(res[1] + res[0] * x)))
    # print("data max@600: ",(y[:-2]-(res[1] + res[0] * x[:-2])))
    print("rms tot (m): ",rms(y-(res[1] + res[0] * x)))
    print("rms max@600 (m): ",rms(y[:]-(res[1] + res[0] * x[:])))
    print("rms max@600 (m): ",rms(y-(deg2fit[2] + deg2fit[1] * x + deg2fit[0] * x**2)))

    ax1.plot(x, res[1] + res[0] * x, 'r-',label='y={:.2f}x+{:.2f} +- std={:.2f}'.format(res[0],res[1],res[-1]))
    ax1.plot(x, (res[1] + res[0] * x)*(1+res[-1]), 'r--')
    ax1.plot(x, (res[1] + res[0] * x)*(1-res[-1]), 'r--')
    ax1.legend()
    ax1.plot(x, y, 'b*')
    ax1.plot(x, (deg2fit[2] + deg2fit[1] * x + deg2fit[0] * x**2), 'g--')
    # ax1.plot(x, y-(deg3fit[3] + deg3fit[2] * x + deg3fit[1] * x**2 + deg3fit[0] * x**3), 'bx')
    # ax1.loglog()

    fig.savefig(
        XovOpt.get("tmpdir") + 'test_roughn.png')
	
	
if __name__ == '__main__':

    sol = "tp6_0"
    exp_list = ['3res_5amp','3res_10amp','3res_20amp','3res_30amp','3res_40amp','3res_60amp','3res_80amp']
    
    fit_sols(sol, exp_list)
