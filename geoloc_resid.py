#!/usr/bin/env python3
# ----------------------------------
# Check discrepancies of geolocalised data elevation w.r.t. DEM
# ----------------------------------
# Author: Stefano Bertone
# Created: 17-Jun-2019
#

import glob
import os
import subprocess
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from ground_track import gtrack
from prOpt import parallel, SpInterp, new_gtrack, outdir, auxdir, local, vecopts, debug
import AccumXov as xovacc
import seaborn as sns

if __name__ == '__main__':

    start = time.time()

    #epo = '1501'
    spk = ['KX', 'AG']

    for ex in spk:

        #print(epo)
        path = "/att/nobackup/sberton2/MLA/out/mladata_"+ex+"/gtrack_*/*.pkl"
        allFiles = glob.glob(os.path.join(path))
        print("Processing ",ex)
        #print(os.path.join(path))
        #print(allFiles)

        track = gtrack(vecopts)
        df = []
        # Prepare list of tracks to geolocalise
        for fil in allFiles:
            track = track.load(fil)
            # print(fil)
            if len(track.ladata_df)>0:
                track.ladata_df = track.ladata_df[['ET_TX', 'TOF', 'chn', 'orbID', 'seqid', 'LON', 'LAT', 'R']]
                df.append(track.ladata_df)

        df = pd.concat(df).reset_index().sort_values(by='ET_TX')
        print("ntracks",len(df))

        gmt_in = 'gmt_' + track.name + '.in'
        if os.path.exists('tmp/' + gmt_in):
            os.remove('tmp/' + gmt_in)
        np.savetxt('tmp/' + gmt_in, list(zip(df.LON.values, df.LAT.values)))

        #dem = auxdir + 'MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
        dem = '/att/nobackup/emazaric/MESSENGER/data/GDR/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
	# r_dem = np.loadtxt('tmp/gmt_' + self.name + '.out')

        r_dem = subprocess.check_output(
            ['grdtrack', gmt_in, '-G' + dem],
            universal_newlines=True, cwd='tmp')
        r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]
        r_dem *= 1.e3

        df['altdiff_dem'] = df.R.values - r_dem
        print(len(df.R.values),len(r_dem[np.abs(r_dem)>0]))
        print(df['altdiff_dem'].values)

        if False:
            ax = plt.gca()
            df.reset_index().plot(x='ET_TX', y='altdiff_dem', ax=ax)
            # ax = df[['altdiff_dem']].plot.hist(bins=2500, alpha=0.5)

            ax.legend()
            ax.set_title('elev_vs_dem')
            # ax.set_xlabel('time')
            # ax.set_ylabel('meters')
            # ax.set_ylim([-3000,3000])
        else:

            fig, ax1 = plt.subplots(1,1)
            plt.hist(df['altdiff_dem'].values, bins=50, density=True, range=(-1500, 1500))  # arguments are passed to np.histogram

            # mu = 0
            # variance = (20) ** 2
            # sigma = math.sqrt(variance)
            #
            mu = df['altdiff_dem'].median(axis=0)
            sorted = np.sort(abs(df['altdiff_dem'].values - mu))
            # print(len(sorted))
            std_median = sorted[round(0.68 * len(sorted))]
            # sorted = sorted[sorted<3*std_median]

            print("Mean, std of data:", df['altdiff_dem'].mean(axis=0), df['altdiff_dem'].std(axis=0))
            print("Median, std_med of data:", mu, std_median)
            print("Max diff: ",df['altdiff_dem'].abs().max())

            x = np.linspace(mu - 3 * std_median, mu + 3 * std_median, 100)
            plt.plot(x, stats.norm.pdf(x, mu, std_median))

            # x1 = df.loc[(df.altdiff_dem<300) & (df.altdiff_dem>-300),'altdiff_dem']
            #
            # # Plot
            # kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})
            #
            # plt.figure(figsize=(10, 7), dpi=80)
            # sns.distplot(x1, color="dodgerblue", label="elev_vs_dem", **kwargs)
            # # plt.xlim(50, 75)
            # plt.legend()
            ax1.set_title('elev_vs_dem: mu={}, sigma={}, max={}'.format(mu,std_median,df['altdiff_dem'].abs().max()))
            # plt.savefig('elev_vs_dem.png')
            plt.savefig('elev_vs_dem_'+ex+'_5y.png')

    # stop clock and print runtime
    # -----------------------------
    end = time.time()
    print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
