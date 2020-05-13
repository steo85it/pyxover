import itertools as itert

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import AccumXov as xovacc
import xov_utils
from xov_setup import xov


def xovnum_plot():
    vecopts = {}

    if False:
        # xov_cmb = xov.load_combine('/home/sberton2/Works/NASA/Mercury_tides/out/sim_mlatimes/0res_1amp/',vecopts)
        xov_cmb = xov_utils.load_combine('/att/nobackup/sberton2/MLA/out/mladata/', vecopts)

        mon = [i for i in range(1, 13)]
        yea = [i for i in range(11, 16)]
        monyea = [str("{:02}".format(y)) + str("{:02}".format(x)) for x in mon for y in yea]

        monyea2 = itert.combinations_with_replacement(monyea, 2)

        df_ = pd.DataFrame(xov_cmb.xovers.loc[:, 'orbA'].astype(str).str[0:4], columns=['orbA'])
        df_['orbB'] = xov_cmb.xovers.loc[:, 'orbB'].astype(str).str[0:4]
        df_ = df_.groupby(['orbA', 'orbB']).size().reset_index().rename(columns={0: 'count'})

        xov_ = xov(vecopts)
        xov_.xovers = df_.copy()
        xov_.save('tmp/df_pltxovmonth.pkl')
    else:
        xov_ = xov(vecopts)
        xov_ = xov_.load('tmp/df_pltxovmonth.pkl')
        print("File loaded...")
        df_ = xov_.xovers.copy()
        print(df_.loc[df_['count'].idxmax()])

    # create pivot table, days will be columns, hours will be rows
    piv = pd.pivot_table(df_, values="count", index=["orbA"], columns=["orbB"], fill_value=0)
    # plot pivot table as heatmap using seaborn

    fig, ax1 = plt.subplots(nrows=1)
    ax1 = sns.heatmap(piv, square=False, annot=False)
    # , robust=True,
    #               cbar_kws={'label': 'RMS (m)'}, xticklabels=piv.columns.values.round(2), fmt='.4g')
    ax1.set(xlabel='orbA',
            ylabel='orbB')
    plt.tight_layout()
    plt.savefig('tmp/xover_month.png')
    plt.close()


if __name__ == '__main__':
    xovnum_plot()
