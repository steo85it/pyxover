# ====================================================================
#
# Name:       plotResRMS.py
#
# Purpose:    Plot RMS of residuals geographically
#
# Author:     D. Arnold
#
# Created:    08-Sep-2015
#
# Changes:    SB 27/04/17 : adapt to ORBDIFF SCH residuals
#
# ====================================================================
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap


# import basemap as Basemap

def mapcount(et, lat, lon):
    # Restrict to particular local times
    # print(lat)
    # print(lon)

    # Define grid
    lonGrid = np.arange(-180, 180 + 1, 1)
    latGrid = np.arange(-90, 91 + 1, 1)
    lonMesh, latMesh = np.meshgrid(lonGrid, latGrid)

    count = np.zeros((latGrid.size, lonGrid.size))

    epo = np.around(et, decimals=5)

    # Round to full degree
    lat = np.around(lat, decimals=0)
    lon = np.around(lon, decimals=0)

    # Map longitudes into interval -180deg to 180deg
    lon[lon > 180] = lon[lon > 180] - 360

    # Fill grid (element 0,0 --> -90deg lat,-180deg lon)
    for i in range(0, lat.size):
        count[int(lat[i]) + 90, int(lon[i]) + 180] += 1

    # Plot

    ### Polar Stereographic projection
    m = Basemap(projection='npstere', boundinglat=10, lon_0=270, resolution='l')
    # m.drawcoastlines()
    # m.fillcontinents(color='coral',lake_color='aqua')
    # draw parallels and meridians.
    m.drawparallels(np.arange(-80., 81., 20.), color='0.5', labels=[True, True, True, True], fontsize=14)
    m.drawmeridians(np.arange(-180., 181., 20.), color='0.5', labels=[True, True, True, True], fontsize=14)
    # m.drawmapboundary(fill_color='aqua')
    # draw tissot's indicatrix to show distortion.
    # ax = plt.gca()
    # for y in np.linspace(m.ymax/20,19*m.ymax/20,10):
    #    for x in np.linspace(m.xmax/20,19*m.xmax/20,10):
    #        lon, lat = m(x,y,inverse=True)
    #        poly = m.tissot(lon,lat,2.5,100,\
    #                        facecolor='green',zorder=10,alpha=0.5)
    x, y = m(lonMesh, latMesh)
    datamap = m.pcolor(x, y, count, vmax=100)
    # plt.title("North Polar Stereographic Projection")
    plt.savefig('testMap.png')

    ### Mollweide or hammer projection
    # m = Basemap(projection='moll', lon_0=0)
    m.warpimage(
        image="/home/sberton2/Works/NASA/Mercury_tides/PyXover/img/Mercury_MESSENGER_MDIS_Basemap_LOI_Mosaic_Global_1024.jpg")
    # m.drawparallels(np.arange(-90,90,30),color='0.5',labels=[False,True,True,False],fontsize=14)
    # x, y = m(lonMesh, latMesh)
    # datamap = m.pcolor(x, y, count, vmin=0.0, vmax=0.5,alpha=0.2)
    ###

    ### Cartesian plot
    # datamap= plt.pcolor(lonMesh,latMesh,count)
    ###

    cmap = plt.get_cmap('jet')

    cb = plt.colorbar(datamap, shrink=0.7, orientation='horizontal', pad=0.05)
    cb.ax.tick_params(labelsize=10)
    cb.set_label("count", fontsize=20)
    # plt.title(title,y=1.08)
    # plt.clim(0,10)

    # plt.colorbar(p,orientation='vertical')
    # plt.show()
    plt.savefig('testMap.png')
    #  plt.show()
    plt.clf()
    plt.close()
