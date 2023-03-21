
# read orbits 0801141902, 0810060836, 1104010231, 1204011915, 1304010004, 1404011002, 1504012318
# store data for geolocation + latlon from MLARDR
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy as spice
from scipy.constants import c as clight

from src.config import XovOpt
from src.pygeoloc.PyGeoloc import launch_gtrack
from src.pygeoloc.ground_track import gtrack
from src.config import XovOpt
from src.xovutil import astro_trans as astr

# update paths and check options
from src.xovutil.project_coord import project_stereographic

XovOpt.set("basedir", 'data/')
XovOpt.set("instrument", 'MLA')
XovOpt.set("local", True)
XovOpt.set("parallel", False)
XovOpt.set("partials", False)
XovOpt.set("expopt", 'PAM')

XovOpt.set("new_gtrack", 2)
vecopts = XovOpt.get('vecopts')
# vecopts['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]
vecopts['ALTIM_BORESIGHT'] = [2.2104999983228e-3, 2.9214999977833e-3, 9.9999328924124e-1]
vecopts['SCFRAME'] = 'MSGR_SPACECRAFT'
# vecopts['PLANETRADIUS'] = 2439.4

XovOpt.set('vecopts', vecopts)
XovOpt.set("SpInterp", 0)
XovOpt.check_consistency()
XovOpt.display()

spice.furnsh(f'{XovOpt.get("auxdir")}mymeta')

version = 2001 # 2101
mod_detail = 17

tracks = []
# radius issue on 1504012318 (did someone switch to 2439.7km at some point?)
# tracknames = ['0801141902', '0810060836', '1104010231', '1204011915', '1304010004', '1404011002', '1504012318']
if version == 2001:
    tracknames = ['0801141902', '0810060836', '1104010231', '1204011915', '1304010004', '1404011002']
else:
    tracknames = ['1104010231', '1204011915', '1304010004', '1404011002']

allFiles = [f"data/raw/{version}/mlascirdr{f}.tab" for f in tracknames][:]
print(list(zip(tracknames, allFiles)))

for track_id, infil in zip(tracknames, allFiles):
    track = track_id
    print(track_id)
    track = gtrack(XovOpt)
    # Read and fill
    track.prepro(infil, read_all=True)
    tracks.append(track)

# print(tracks)
# print(XovOpt.get("pert_cloop"))
# exit()
for track in tracks:
    print(f"Processing {track.name}...")
    # extract emission, reception and travel time from RDR
    inp_df = track.ladata_df
    tof = inp_df['TOF'].values
    et_tx = inp_df['ET_TX'].values
    et_rx = et_tx + tof

    # get SC to SSB at TX/RX
    for idx, et in enumerate([et_tx,et_rx]):
        scpv, lt = spice.spkezr(vecopts['SCNAME'],
                               et,
                               vecopts['INERTIALFRAME'],
                               'NONE',
                               vecopts['INERTIALCENTER'])
        scpv = np.vstack(scpv)
        if idx==0:
            scpos_tx = 1.e3 * scpv[:, :3]
            scvel_tx = 1.e3 * scpv[:, 3:]
        else:
            scpos_rx = 1.e3 * scpv[:, :3]
            scvel_rx = 1.e3 * scpv[:, 3:]

    # get norms
    Rtx = np.linalg.norm(scpos_tx, axis=1)
    Rrx = np.linalg.norm(scpos_rx, axis=1)

    # get altimeter boresight in S/C frame
    zpt = np.tile(vecopts['ALTIM_BORESIGHT'], np.size(scpos_tx, 0)).reshape(-1, 3)
    # compute s/c frame to inertial rotation
    pxform_array = np.frompyfunc(spice.pxform, 3, 1)
    cmat = pxform_array(vecopts['SCFRAME'], vecopts['INERTIALFRAME'], et_tx)
    # rotate boresight dir to inertial frame
    zpt = [np.dot(cmat[i], zpt[i]) for i in range(0, np.size(zpt, 0))]
    e1 = np.vstack(zpt)

    rin = {}
    ##############################
    # SM
    ##############################
    scpv_pla, lt = spice.spkezr(vecopts['SCNAME'],
                            et_tx,
                            vecopts['INERTIALFRAME'],
                            'NONE',
                            vecopts['PLANETID'])
    rs = 1.e3 * np.vstack(scpv_pla)[:, :3]
    vs = 1.e3 * np.vstack(scpv_pla)[:, 3:]

    r1 = (clight * tof / 2.)[:,np.newaxis] * e1
    rin[5] = rs + r1

    ##############################
    # SMM
    ##############################
    # aberration stuff
    r12 = scpos_rx-scpos_tx
    v12 = r12/tof[:,np.newaxis]
    # e12 = r12/np.linalg.norm(r12,axis=1)[:,np.newaxis]
    # v12 = np.einsum('ij,ij->i',vs,e12)[:,np.newaxis]*e12   # use sc velocity relative to planet proj on r12
    beta = v12/clight
    betanorm = np.linalg.norm(beta, axis=1)
    costheta = np.einsum('ij,ij->i',v12,e1)/np.linalg.norm(v12,axis=1)
    # costheta1 = np.einsum('ij,ij->i',r12,e1)/np.linalg.norm(r12,axis=1)
    # assert np.max(np.abs(costheta1-costheta)) < 1.e-10
    # print(np.shape(beta), np.shape(costheta))

    # get planet barycenter state (SSB J2000) at bounce
    # --------------------------------------------------
    et_bc = et_tx + (tof / 2.) * (1+betanorm*costheta)
    # print(np.max(np.abs(et_tx + tof / 2.-et_bc)))

    plapos_bc, lt = spice.spkpos(vecopts['PLANETNAME'],
                                 et_bc,
                                 vecopts['INERTIALFRAME'],
                                 'NONE',
                                 vecopts['INERTIALCENTER'])

    plapos_bc = 1.e3 * np.vstack(plapos_bc)

    tmp = clight * tof/2. * (1+betanorm*costheta)
    rin[17] = tmp[:,np.newaxis] * e1 + scpos_tx - plapos_bc

    ##############################
    # PAM
    ##############################
    tmp = clight * tof/2. * (1-betanorm*betanorm*(1-costheta*costheta))
    rin[24] = tmp[:,np.newaxis] * (e1 + beta) + scpos_tx - plapos_bc

    # Choose level of detail and plot diffs
    ##########################################
    rin = rin[mod_detail].copy()

    # to mercury-fixed frame
    tsipm = pxform_array(vecopts['INERTIALFRAME'], vecopts['PLANETFRAME'], et_bc)
    rin_bf = [np.dot(tsipm[i], rin[i]) for i in range(0, np.size(rin, 0))]

    rtmp, lattmp, lontmp = astr.cart2sph(rin_bf)
    lat = np.rad2deg(lattmp)
    lon = np.rad2deg(lontmp)
    X, Y = project_stereographic(lon, lat, 0, 90, vecopts['PLANETRADIUS']*1e3)

    R = rtmp

    # if version == 2001:
    #     lonlatr_cols = ['Radius', 'Latitude', 'Longitude']
    # else:
    lonlatr_cols = ['altitude', 'geoc_lat', 'geoc_long']

    tmp_vec_RDR=inp_df[lonlatr_cols].values
    X_rdr, Y_rdr = project_stereographic(tmp_vec_RDR[:,2], tmp_vec_RDR[:,1], 0, 90, vecopts['PLANETRADIUS']*1e3)
    dx = X - X_rdr
    dy = Y - Y_rdr
    dxy = np.sqrt(np.power(dx,2)+np.power(dy,2))
    R_rdr = tmp_vec_RDR[:,0] * 1.e3
    dr = R - R_rdr

    if track.name in ['0801141902', '0810060836']:
        diff = pd.DataFrame(np.vstack([track.ladata_df.loc[:,'geoc_long'].values,dxy,dr]).T, columns=["lon","dxy","dr"])
        diff.plot(x="lon", y=["dxy","dr"])
    else:
        diff = pd.DataFrame(np.vstack([track.ladata_df.loc[:,'geoc_lat'].values,dxy,dr]).T, columns=["lat","dxy","dr"])
        diff.plot(x="lat", y=["dxy","dr"])
    plt.title(track.name)
    plt.savefig(f"out/plt/drdr_{track.name}_{version}_{mod_detail}.png")
    plt.show()
