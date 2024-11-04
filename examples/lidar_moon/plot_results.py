import numpy as np
import pandas as pd
import spiceypy as sp
from matplotlib import pyplot as plt
import seaborn as sns
import glob

#basedir = "/home/sberton2/Works/NASA/Mercury_tides/pyxover_release/examples/lidar_moon/out/2H/cnt/"
basedir = "/att/nobackup/sberton2/PAWSTEL/PyXover/examples/lidar_moon/out/2H/per/"

if __name__ == '__main__':

    csv_files = glob.glob(f"{basedir}df_results*.csv")
    dfl = []
    for csv_name in csv_files:
        df = pd.read_csv(csv_name,index_col=0)
        df.h2_err *= 100.
        dfl.append(df)
    df = pd.concat(dfl)
#    print(df)
#    exit()
        
    fig, sub = plt.subplots(2, 2)

    vmax = df.quantile(0.9,axis=0).h2_err
    vmin = df.quantile(0.1,axis=0).h2_err
    # plt.figure(figsize=(12, 6))
    for idx, doff in enumerate(set(pd.unique(df.doffmax))):
        # print(doff)
        df_ = df.loc[df.doffmax==doff].pivot_table(values='h2_err',index='dlat',columns='dlon')
        # print(df_)
        sns.heatmap(df_, cmap ='RdYlGn_r', linewidths = 0.30, annot = True, ax=sub.flat[idx],
                    vmin=vmin, vmax=vmax, cbar=False)
        sub.flat[idx].set_title(f'doffmax={doff}')
        # sub.flat[idx].set_title(f'doffmax={doff}')
    # fig.colorbar(label='%')
    fig.tight_layout(h_pad=3,w_pad=1)

    plt.suptitle(f"h2 formerr (%) for {basedir.split('/')[-3]}-{basedir.split('/')[-2]}\n")
    plt.subplots_adjust(top=0.87)
    # plt.show()
    plt.savefig(f"{basedir}results.png", dpi=1200)
