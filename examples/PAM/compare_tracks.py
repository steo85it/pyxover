import matplotlib.pyplot as plt
import numpy as np

from src.config import XovOpt
from src.pygeoloc.ground_track import gtrack

XovOpt.display()
XovOpt.set("debug", False)
XovOpt.set("basedir", 'data/')
XovOpt.set("instrument", 'MLA')
XovOpt.set("local", True)
XovOpt.set("parallel", False)
XovOpt.set("partials", False)
XovOpt.set("expopt", 'PAM')
XovOpt.set("monthly_sets", True)

XovOpt.set("new_gtrack", 2)
vecopts = XovOpt.get('vecopts')
vecopts['ALTIM_BORESIGHT'] = [2.2104999983228e-3, 2.9214999977833e-3, 9.9999328924124e-1]
vecopts['SCFRAME'] = 'MSGR_SPACECRAFT'
XovOpt.set('vecopts', vecopts)
XovOpt.set("SpInterp", 0)
XovOpt.set("compute_input_xov", True)

XovOpt.check_consistency()

version = 2001
if version == 2001:
    tracknames = ['0801141902', '0810060836', '1104010231', '1204011915', '1304010004', '1404011002']
else:
    tracknames = ['1104010231', '1204011915', '1304010004', '1404011002']

dist_list = []
for trid in tracknames:
    track_smm = f"data/out/sim/SMM_0/0res_1amp/gtrack/gtrack_{trid}.pkl"
    track_pam = f"data/out/sim/PAM_0/0res_1amp/gtrack/gtrack_{trid}.pkl"


    gtr_smm = gtrack(XovOpt.to_dict())
    gtr_pam = gtrack(XovOpt.to_dict())

    gtr_smm = gtr_smm.load(track_smm)
    gtr_pam = gtr_pam.load(track_pam)

    # print(gtr_smm.ladata_df)
    # print(gtr_pam.ladata_df)

    merge = gtr_smm.ladata_df[['ET_TX','X_stgprj','Y_stgprj']].merge(
                gtr_pam.ladata_df[['ET_TX','X_stgprj','Y_stgprj']], on = 'ET_TX', how = 'left')
    # print(merge)
    dist = merge.apply(lambda x: np.sqrt((x['X_stgprj_x'] - x['X_stgprj_y'])**2 +
                                        (x['Y_stgprj_x'] - x['Y_stgprj_y'])**2), axis=1)*1.e3
    dist_list.append(dist)

    dist.plot.hist()
    plt.title(f"PAM vs SMM for #{trid}")
    plt.xlabel("meters")
    plt.ylabel("# MLA points")
    plt.show()

total_dist = np.concatenate([x.values for x in dist_list],axis=0)
plt.hist(total_dist) #, bins=100)
plt.title(f"PAM vs SMM for all tracks")
plt.xlabel("meters")
plt.ylabel("# MLA points")
plt.show()
# maybe plot dist(lat)