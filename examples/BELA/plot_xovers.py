import glob
import time

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import shapely

from src.config import XovOpt
from src.pygeoloc.ground_track import gtrack
from src.pyxover.xov_setup import xov

# define crs
crs_lonlat = "+proj=lonlat +units=m +a=2440.e3 +b=2440.e3 +no_defs"
crs_stereo_km = '+proj=stere +lat_0=90 +lon_0=0 +lat_ts=90 +k=1 +x_0=0 +y_0=0 ' \
                '+units=km +a=2440.e3 +b=2440.e3 +no_defs'

# read xov list
xov_ = xov(XovOpt.to_dict())
xov_list = [xov_.load(x) for x in
            glob.glob("data/out/sim/BE0_0/3res_20amp/xov/xov_26*_26*.pkl")]
xov_cmb = xov(XovOpt.to_dict())
xov_cmb.combine(xov_list)
# print(xov_cmb.xovers)
# print(xov_cmb.xovers.columns)
df0 = xov_cmb.xovers.copy().reset_index(drop=True)  # .values[::msrm_sampl]


def prepare_grid(gdf):
    """
    Gridding GeoDataFrame
    see https://james-brennan.github.io/posts/fast_gridding_geopandas/
    @param gdf: input GeoDataFrame (used for bounds and crs)
    @return: cells, gdf
    """

    # total area for the grid
    xmin, ymin, xmax, ymax = gdf.total_bounds
    # how many cells across and down
    n_cells = 100
    cell_size = (xmax - xmin) / n_cells
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax + cell_size, cell_size):
        for y0 in np.arange(ymin, ymax + cell_size, cell_size):
            # bounds
            x1 = x0 - cell_size
            y1 = y0 + cell_size
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
    return gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=gdf.crs)


# split xovers in lowlat and polar ones
dfs = []
dfs.append(df0.loc[df0.LAT < 85][:])
dfs.append(df0.loc[(df0.LAT >= 85) & (df0.LAT < 89)][:])
dfs.append(df0.loc[df0.LAT >= 89][:int(2e6)])

# set image
fig = plt.figure()
gs = GridSpec(2, 2, figure=fig)
plt.suptitle("# of crossovers on grid")

for idx, df in enumerate(dfs):

    start = time.time()

    # xov coordinates to geopandas (loooong when many xovs...)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.LON, df.LAT), crs=crs_lonlat)

    # if polar xovers, project to stereo
    if idx == 0:
        ax = fig.add_subplot(gs[0, :])
        vmax = 500
    elif idx == 1:
        ax = fig.add_subplot(gs[1, 0])
        gdf = gdf.to_crs(crs_stereo_km)
        vmax = 250
    elif idx == 2:
        ax = fig.add_subplot(gs[1, 1])
        gdf = gdf.to_crs(crs_stereo_km)
        vmax = 2000

    # prepare grid to plot stats
    cell = prepare_grid(gdf)

    # merge gdfs
    merged = gpd.sjoin(gdf, cell, how='left', op='within')

    # make a simple count variable that we can sum
    merged['n_xov'] = 1
    # Compute stats per grid cell -- aggregate xovs to grid cells with dissolve
    dissolve = merged.dissolve(by="index_right", aggfunc="count")
    # put this into cell
    cell.loc[dissolve.index, 'n_xov'] = dissolve.n_xov.values

    # plot number of xovs per cell
    cell.plot(column='n_xov',
              # figsize=(12, 8),
              cmap='viridis',
              vmax=vmax,
              # edgecolor="grey",
              legend=True, ax=ax)

    print(f"- Plotting {len(df)} xovers in "
          f"{round(time.time() - start,2)} seconds...")

    # add stereo axes labels (TODO should use ax.set_...)
    # if idx > 0:
    #     plt.xlabel("x distance from NP, km")
    #     plt.ylabel("y distance from NP, km")

plt.savefig('data/tmp/xovers_density.pdf')
plt.show()

