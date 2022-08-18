
# read orbits 0801141902, 0810060836, 1104010231, 1204011915, 1304010004, 1404011002, 1504012318
# store data for geolocation + latlon from MLARDR
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy as spice

from config import XovOpt
from geolocate_altimetry import geoloc
from pygeoloc.PyGeoloc import launch_gtrack
from pygeoloc.ground_track import gtrack
from config import XovOpt

# update paths and check options
from xovutil.project_coord import project_stereographic

XovOpt.set("basedir", 'data/')
XovOpt.set("instrument", 'MLA')
XovOpt.set("local", True)
XovOpt.set("parallel", False)
XovOpt.set("partials", False)
XovOpt.set("expopt", 'PAM')

XovOpt.set("new_gtrack", 2)
vecopts = XovOpt.get('vecopts')
vecopts['ALTIM_BORESIGHT'] = [2.2104999983228e-3, 2.9214999977833e-3, 9.9999328924124e-1]
vecopts['SCFRAME'] = 'MSGR_SPACECRAFT'
XovOpt.set('vecopts', vecopts)
XovOpt.set("SpInterp", 0)
XovOpt.check_consistency()
XovOpt.display()

spice.furnsh(f'{XovOpt.get("auxdir")}mymeta')

version = 2001 # 2101
use_gtrack = True

tracks = []
if version == 2001:
    tracknames = ['0801141902', '0810060836', '1104010231', '1204011915', '1304010004', '1404011002']
else:
    tracknames = ['1104010231', '1204011915', '1304010004', '1404011002']

allFiles = [f"data/raw/{version}/mlascirdr{f}.tab" for f in tracknames][:]
print(list(zip(tracknames, allFiles)))

for track_id, infil in zip(tracknames, allFiles):
    track = track_id
    print(track_id)
    track = gtrack(XovOpt.to_dict())
    # Read and fill
    track.prepro(infil, read_all=True)
    tracks.append(track)

# print(tracks)
# print(XovOpt.get("pert_cloop"))
# exit()

for track in tracks:

    if use_gtrack:
        track.setup(f"data/raw/{version}/mlascirdr{track.name}.tab")
        inp_df = track.ladata_df.copy()
        X_geoloc, Y_geoloc = inp_df[['X_stgprj', 'Y_stgprj']].values.T*1e3
        R_geoloc = vecopts['PLANETRADIUS'] * 1e3 + inp_df.loc[:, 'R'].values
    else:
        inp_df = track.ladata_df.copy()
        tmp_pertPar = track.pertPar.copy()
        tmp_pertPar = tmp_pertPar.fromkeys(tmp_pertPar, 0)
        results = geoloc(inp_df, vecopts, tmp_pertPar=tmp_pertPar, SpObj=[], t0=0)
        X_geoloc, Y_geoloc = project_stereographic(results[0][:, 0], results[0][:, 1], 0, 90, vecopts['PLANETRADIUS']*1.e3)
        R_geoloc = results[0][:, 2]

    lonlatr_cols = ['altitude', 'geoc_lat', 'geoc_long']
    tmp_vec_RDR=inp_df[lonlatr_cols].values
    X_rdr, Y_rdr = project_stereographic(tmp_vec_RDR[:,2], tmp_vec_RDR[:,1], 0, 90, vecopts['PLANETRADIUS']*1e3)
    R_rdr = tmp_vec_RDR[:,0] * 1.e3

    dx = X_geoloc - X_rdr
    dy = Y_geoloc - Y_rdr
    dxy = np.sqrt(np.power(dx,2)+np.power(dy,2))
    dr = R_geoloc - R_rdr

    if track.name in ['0801141902', '0810060836']:
        diff = pd.DataFrame(np.vstack([track.ladata_df.loc[:,'geoc_long'].values,dxy,dr]).T, columns=["lon","dxy","dr"])
        diff.plot(x="lon", y=["dxy","dr"])
    else:
        diff = pd.DataFrame(np.vstack([track.ladata_df.loc[:,'geoc_lat'].values,dxy,dr]).T, columns=["lat","dxy","dr"])
        diff.plot(x="lat", y=["dxy","dr"])
    plt.title(track.name)
    pltfil = f'out/plt/pyx/drdr_{track.name}_{version}.png'
    plt.savefig(pltfil)
    print(f"- Geoloc residuals saved to {pltfil}")
    #    plt.show()
