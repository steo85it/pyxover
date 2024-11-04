import glob
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import XovOpt
from pygeoloc.ground_track import gtrack
from pyxover.xov_setup import xov

# xov_path = "examples/BELA/data/out/sim/BE0_0/3res_20amp/xov/xov_2612_2612.pkl"
xov_ = xov(XovOpt.to_dict())
xov_list = [xov_.load(x) for x in
            glob.glob("data/out/sim/BE0_0/3res_20amp/xov1/xov_26*_26*.pkl")]
xov_cmb = xov(XovOpt.to_dict())
xov_cmb.combine(xov_list)
print(xov_cmb.xovers)
print(xov_cmb.xovers.columns)
print(xov_cmb.xovers.LAT.min())
xov_cmb.xovers = xov_cmb.xovers.loc[xov_cmb.xovers.LAT < 70].reset_index(drop=True)

print(xov_cmb.xovers)
#exit()

for idxov in tqdm(range(len(xov_cmb.xovers))[:5]):
    orbidA, orbidB, shotidA, shotidB = xov_cmb.xovers.loc[idxov,['orbA','orbB',
                                                             'mla_idA','mla_idB']].values

    crs_lonlat = "+proj=lonlat +units=m +a=2440.e3 +b=2440.e3 +no_defs"
    crs_stereo_km = '+proj=stere +lat_0=90 +lon_0=0 +lat_ts=90 +k=1 +x_0=0 +y_0=0 ' \
                    '+units=km +a=2440.e3 +b=2440.e3 +no_defs'
    ax = plt.subplot()
    for orbid, shotid in zip([orbidA, orbidB],[shotidA, shotidB]):
        trackfil = f"data/out/sim/BE0_0/3res_20amp/gtrack_26/gtrack_{orbid}.pkl"
        track = gtrack(XovOpt.to_dict())
        track = track.load(trackfil)
        ladata_df = track.ladata_df
        df0 = ladata_df.loc[ladata_df['orbID'] == orbid]  # .values[::msrm_sampl]
        gdf0 = gpd.GeoDataFrame(
            df0, geometry=gpd.points_from_xy(df0.LON, df0.LAT), crs=crs_lonlat)
        gdf0.to_crs(crs_stereo_km).plot(ax=ax) #, label=gdf0.orbID[0])  # , color='red')

    df0 = xov_cmb.xovers.copy()  # .values[::msrm_sampl]
    gdf0 = gpd.GeoDataFrame(
        df0, geometry=gpd.points_from_xy(df0.LON, df0.LAT), crs=crs_lonlat)
    x1, y1 = list(zip(*gdf0.to_crs(crs_stereo_km).geometry.values[0].coords.xy))[0]
    # x0, y0 = xov_cmb.xovers.loc[0,['x0','y0']].values
    # print(x1, y1, x0, y0)
    # plt.scatter(x=x0, y=y0)
    plt.scatter(x=x1, y=y1)
    plt.xlim(x1-50.,x1+50.)
    plt.ylim(y1-50.,y1+50.)
    plt.xlabel("km (x, NP stereo proj)")
    plt.ylabel("km (y, NP stereo proj)")
    plt.savefig(f"plt/tst_{idxov}.png")
    plt.show()
