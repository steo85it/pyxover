import datetime

import numpy as np
import pandas as pd
import spiceypy as spice

from xovutil import astro_trans as astr
from config import XovOpt

def prepro_ilmNG(illumNGf):
    li = []
    for f in illumNGf:
        print("Processing", f)
        df = pd.read_csv(f, index_col=None, header=0, names=[f.split('.')[-1]])
        li.append(df)

    # df_ = dfin.copy()
    df_ = pd.concat(li, axis=1)
    df_ = df_.apply(pd.to_numeric, errors='coerce')
    # print(df_.rng.min())

    df_ = df_[df_.rng < 1600]
    df_ = df_.rename(columns={"xyzd": "epo_tx"})
    # print(df_.dtypes)

    df_['diff'] = df_.epo_tx.diff().fillna(0)
    # print(df_[df_['diff'] > 1].index.values)
    arcbnd = [df_.index.min()]
    # new arc if observations separated by more than 1h
    arcbnd.extend(df_[df_['diff'] > 3600].index.values)
    arcbnd.extend([df_.index.max() + 1])
    # print(arcbnd)
    df_['orbID'] = 0
    for i, j in zip(arcbnd, arcbnd[1:]):
        orbid = (datetime.datetime(2000, 1, 1, 12, 0) + datetime.timedelta(seconds=df_.loc[i, 'epo_tx'])).strftime(
            "%y%m%d%H%M")
        df_.loc[df_.index.isin(np.arange(i, j)), 'orbID'] = orbid

    return df_


def prepro_BELA_sim(epo_in):
    try:
        scpv, lt = spice.spkezr(XovOpt.get("vecopts")['SCNAME'],
                               epo_in,
                               XovOpt.get("vecopts")['PLANETFRAME'],
                               'LT',
                               XovOpt.get("vecopts")['PLANETNAME'])
    except:
        scpv = np.array([spice.spkez(XovOpt.get("vecopts")['SCID'],
                                t,
                                XovOpt.get("vecopts")['PLANETFRAME'],
                                'LT',
                                XovOpt.get("vecopts")['PLANETID'])[0] for t in epo_in])

    scpos = np.array(scpv)[:,:3]
    range = np.linalg.norm(scpos,axis=1) - XovOpt.get("vecopts")['PLANETRADIUS']

    scplavec = scpos/np.linalg.norm(scpos,axis=1)[:,None]
    approx_bounce_point = scplavec*XovOpt.get("vecopts")['PLANETRADIUS'] #range[:,None]

    df_ = pd.DataFrame(approx_bounce_point,columns=['x','y','z'])
    df_['epo_tx'] = epo_in
    df_['rng'] = range

    approx_bounce_point_sph = astr.cart2sph(approx_bounce_point)
    df_['lat']= np.rad2deg(approx_bounce_point_sph[1]) # pd.DataFrame(approx_bounce_point_sph,columns=['r','lat','lon'])

    # apply altitude cutoff (PFD too high)
    df_ = df_[df_.rng < 1600]
    df_ = df_.rename(columns={"xyzd": "epo_tx"})
    # print(df_.dtypes)

    ### used for MLA ###
    # df_['diff'] = df_.epo_tx.diff().fillna(0)
    # # print(df_[df_['diff'] > 1].index.values)
    # arcbnd = [df_.index.min()]
    # # new arc if observations separated by more than 1h
    # arcbnd.extend(df_[df_['diff'] > 3600].index.values)

    # for BELA, new arc at every upwards passage of equator
    df_['diff'] = df_.lat.diff().fillna(0)
    # print(df_[(df_['diff'] > 0)])
    # print(df_[(df_['diff'] > 0) & (df_['lat'].round(1) == 0)])

    def ranges(nums):
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        return list(zip(edges, edges))

    # set up arc boundaries (and remove consecutive indexes due to approx)
    arcbnd = [df_.index.min()]
    arcbnd.extend(df_[(df_['diff'] > 0) & (df_['lat'].round(0) == 0)].index.values)
    arcbnd = [x[0] for x in ranges(arcbnd)]
    arcbnd.extend([df_.index.max() + 1])
    # print(arcbnd)

    df_['orbID'] = 0
    for i, j in zip(arcbnd, arcbnd[1:]):
        orbid = (datetime.datetime(2000, 1, 1, 12, 0) + datetime.timedelta(seconds=df_.loc[i, 'epo_tx'])).strftime(
            "%y%m%d%H%M")
        df_.loc[df_.index.isin(np.arange(i, j)), 'orbID'] = orbid

    return df_