import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# from examples.MLA.options import XovOpt.get("tmpdir")
from config import XovOpt


def plt_histo_dR(idx, xov_df, xov_ref='', xlim=None):
    # import scipy.stats as stats
    xov_df = xov_df.loc[xov_df.dR.abs()<200]

    if xlim == None:
        xlim = 50

    plt.figure(figsize=(8,3))
    plt.xlim(-1.*xlim, xlim)
    # the histogram of the data
    num_bins = 200 # 'auto'
    n, bins, patches = plt.hist(xov_df.dR.astype(np.float), bins=num_bins, density=False, facecolor='blue',
                                alpha=0.9, range=[-1.*xlim, xlim])
    # add a 'best fit' line
    # y = stats.norm.pdf(bins, mean_dR, std_dR)
    # plt.plot(bins, y, 'b--')
    if isinstance(xov_ref, pd.DataFrame):
        xov_ref = xov_ref.loc[xov_ref.dR.abs() < 200]
        n, bins, patches = plt.hist(xov_ref.dR.astype(np.float), bins=num_bins, density=False, facecolor='red',
                                    alpha=0.3, range=[-1.*xlim, xlim])
    plt.xlabel('dR (m)')
    plt.ylabel('Probability')

    mean_dR = xov_df.dR.mean()
    std_dR = xov_df.dR.std()
    plt.title(r'Histogram of dR: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig(XovOpt.get("tmpdir") + '/histo_dR_' + str(idx) + '.png')
    plt.clf()
    print("### plt_histo_dR: Plot saved to ", XovOpt.get("tmpdir") + '/histo_dR_' + str(idx) + '.png')


def plt_geo_dR(sol, xov, truncation=None):
    # select only obs with dR<200 meters
    xov_df = xov.xovers.copy()
    if truncation:
        xov_df = xov_df.loc[xov_df.dR.abs()<truncation]
    # dR absolute value taken
    xov_df['dR_orig'] = xov_df.dR.values
    xov_df['dR'] = xov_df.dR.abs()
    # print(xov.xovers.LON.max(),xov.xovers.LON.min())
    mladR = xov_df.round({'LON': 0, 'LAT': 0, 'dR': 3}).groupby(['LON', 'LAT']).dR.median().reset_index()
    # print(mladR.LON.max(),mladR.LON.min())
    # exit()
    fig, ax1 = plt.subplots(nrows=1)
    # ax0.errorbar(range(len(dR_avg)),dR_avg, yerr=dR_std, fmt='-o')
    # ax0.set(xlabel='Exp', ylabel='dR_avg (m)')
    # ax1 = sns.heatmap(mlacount, square=False, annot=True, robust=True)
    # cmap = sns.palplot(sns.light_palette("green"), as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    piv = pd.pivot_table(mladR, values="dR", index=["LAT"], columns=["LON"], fill_value=0)
    # plot pivot table as heatmap using seaborn

    #piv = (piv + empty_geomap_df).fillna(0)
    # print(piv)
    # exit()
    sns.heatmap(piv, xticklabels=10, yticklabels=10)
    plt.ylim(90,-5)
    plt.tight_layout()
    ax1.invert_yaxis()
    #         ylabel='Topog ampl rms (1st octave, m)')
    fig.savefig(XovOpt.get("tmpdir") + '/mla_dR_' + sol + '.png')
    plt.clf()
    plt.close()
    print("### plt_geo_dR: Plot saved to ", XovOpt.get("tmpdir") + '/mla_dR_' + sol + '.png')