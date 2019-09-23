import sys
sys.path.insert(0,'/home/sberton2/projects/Mercury_tides/PyXover_sim')

from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns

from ground_track import gtrack
from prOpt import vecopts, outdir
import matplotlib.pyplot as plt

if __name__ == '__main__':

    sim_test = "sph"

    track_sim = gtrack(vecopts)
    simf = glob("/att/nobackup/sberton2/MLA/data/SIM_13/"+sim_test+"/0res_1amp/MLASIMRDR13010107*.TAB")

    for f in simf:
        track_sim.read_fill(f)

    print(track_sim.ladata_df)


    track_real = gtrack(vecopts)
    reaf = glob("/att/nobackup/sberton2/MLA/data/MLA_13/MLASCIRDR13010107*.TAB")
    for f in reaf:
        track_real.read_fill(f)

    sim_df = track_sim.ladata_df.apply(pd.to_numeric, errors='ignore', downcast='float')
    rea_df = track_real.ladata_df[1:].apply(pd.to_numeric, errors='ignore', downcast='float').reset_index()

    tmp = sim_df['TOF'].subtract(rea_df['TOF'])
    tmp.columns = [f+"_dlt" for f in tmp.columns]
    diff = pd.concat([rea_df,tmp], axis=1)
    print(diff)
    #
    empty_geomap_df = pd.DataFrame(0, index=np.arange(0, 91),
                                   columns=np.arange(-180, 181))

    fig, ax1 = plt.subplots(nrows=1)
    diff = diff.round({'geoc_long': 0, 'geoc_lat': 0, 'TOF_dlt': 10}).groupby(['geoc_long','geoc_lat']).mean().reset_index()
    print(diff)
    # exit()
    # Draw the heatmap with the mask and correct aspect ratio
    piv = pd.pivot_table(diff, values="TOF_dlt", index=["geoc_lat"], columns=["geoc_long"], fill_value=0)
    # plot pivot table as heatmap using seaborn
    piv = (piv + empty_geomap_df).fillna(0)
    # print(piv)
    sns.heatmap(piv, xticklabels=10, yticklabels=10)
    plt.tight_layout()
    ax1.invert_yaxis()
    #         ylabel='Topog ampl rms (1st octave, m)')
    fig.savefig('../tmp/mla_simdiff_' + sim_test + '.png')
    plt.clf()
    plt.close()


    exit()
