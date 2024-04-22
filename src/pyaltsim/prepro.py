import datetime

import numpy as np
import pandas as pd
import spiceypy as spice

from xovutil import astro_trans as astr
from config import XovOpt

def prepro_ilmNG(illumNGf):
    li = []
    for f in illumNGf:
        # print("Processing", f)
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

# First a priori to generate altimetry data. It is generated based on a
# spherical planet, and based on emission time = reception time
# It is later used for iterations on the light time to generate
# realistic altimetry data
def prepro_BELA_sim(epo_in):
    scpv, lt = spice.spkezr(XovOpt.get("vecopts")['SCNAME'],
                               epo_in,
                               XovOpt.get("vecopts")['PLANETFRAME'],
                               'LT',
                               XovOpt.get("vecopts")['PLANETNAME'])

    scpos = np.array(scpv)[:,:3]
    range_val = np.linalg.norm(scpos,axis=1) - XovOpt.get("vecopts")['PLANETRADIUS']

    scplavec = scpos/np.linalg.norm(scpos,axis=1)[:,None]
    approx_bounce_point = scplavec*XovOpt.get("vecopts")['PLANETRADIUS'] #range_val[:,None]

    df_ = pd.DataFrame(approx_bounce_point,columns=['x','y','z'])
    df_['epo_tx'] = epo_in
    df_['rng'] = range_val

    approx_bounce_point_sph = astr.cart2sph(approx_bounce_point)
    df_['lat']= np.rad2deg(approx_bounce_point_sph[1]) # pd.DataFrame(approx_bounce_point_sph,columns=['r','lat','lon'])

    # apply altitude cutoff (PFD too high)
    df_ = df_[df_.rng < XovOpt.get("max_range_altitude")]
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

    def ranges(nums):
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        return list(zip(edges, edges))

    # set up arc boundaries (and remove consecutive indexes due to approx)
    arcbnd = [df_.index.min()]
    arcbnd.extend(df_[(df_['diff'] > 0) & (df_['lat'].round(0) == 0)].index.values)
    ranges_arcbnd = ranges(arcbnd)
    arcbnd = [ranges_arcbnd[0][0]]
    # Look for first sign change in ranges_arcbnd
    for i,j in ranges_arcbnd[1:][:]:
        for k in range(i+1,j):
            if df_.loc[i, 'lat']*df_.loc[k, 'lat']<0:
                arcbnd.append(k)
                break
    # arcbnd = [x[0] for x in ranges(arcbnd)]
    arcbnd.extend([df_.index.max() + 1])

    df_['orbID'] = 0
    for i, j in zip(arcbnd, arcbnd[1:]):
        orbid = (datetime.datetime(2000, 1, 1, 12, 0) + datetime.timedelta(seconds=df_.loc[i, 'epo_tx'])).strftime(
            "%y%m%d%H%M")
        df_.loc[df_.index.isin(np.arange(i, j)), 'orbID'] = orbid

    return df_
