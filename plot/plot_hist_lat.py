import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_path = "/storage/research/aiub_gravdet/WD_BELA/"
path = "/home/wdesprat/nobackup/pyxover/plot/"

pyout_folder = f"{data_path}pyXover/out/"
iter = 0

id_ = ["CB9","AE2","AE3","BB3"]
if False:
   all_lat=dict()
   for id in id_:
      filenames = glob.glob(f"{pyout_folder}{id}_{iter}/xov/xov_*.pkl")

      all_lat[id] = np.concatenate([
         pd.read_pickle(f).xovers["LAT"].to_numpy()
         for f in filenames
         ])

   all_lat_ext=dict()
   for id in id_[1:]:
      filenames = glob.glob(f"{pyout_folder}{id}_{iter}/xov_ext/xov_*.pkl")

      all_lat_ext[id] = np.concatenate([
         pd.read_pickle(f).xovers["LAT"].to_numpy()
         for f in filenames
         ])

   np.savez("lat_cache.npz",
            all_lat=all_lat,
            all_lat_ext=all_lat_ext)
else:
   cache = np.load(path+"lat_cache.npz", allow_pickle=True)

   all_lat = cache["all_lat"].item()
   all_lat_ext = cache["all_lat_ext"].item()

nbins=50
fig = plt.figure()
# datasets = {
#     "MLA/MLA (N)": all_lat[id_[0]],
#     "BELA/BELA (N)": all_lat[id_[1]],
#     # "BELA/BELA (S)": np.abs(all_lat[id_[2]]),
#     "MLA/BELA (N)": all_lat[id_[3]],
# 
#     "BELA/BELA ext (N)": np.concatenate([all_lat[id_[1]], all_lat_ext[id_[1]]]),
#     # "BELA/BELA ext (S)": np.abs(np.concatenate([all_lat[id_[2]], all_lat_ext[id_[2]]])),
#     "MLA/BELA ext (N)": np.concatenate([all_lat[id_[3]], all_lat_ext[id_[3]]]),
# }
datasets = {
    "MLA/MLA (N)": all_lat[id_[0]],
    "BELA/BELA (N)": np.concatenate([all_lat[id_[1]], all_lat_ext[id_[1]]]),
    "BELA/BELA (S)": np.abs(np.concatenate([all_lat[id_[2]], all_lat_ext[id_[2]]])),
    "MLA/BELA (N)": np.concatenate([all_lat[id_[3]], all_lat_ext[id_[3]]]),
}

# datasets = {
#     "MLA/MLA": all_lat[id_[0]],
#     "BELA/BELA": np.concatenate([all_lat[id_[1]],all_lat[id_[2]]]),
#     "MLA/BELA": all_lat[id_[3]],
#     "BELA/BELA ext": np.concatenate([all_lat[id_[1]], all_lat_ext[id_[1]], all_lat[id_[2]], all_lat_ext[id_[2]]]),
#     "MLA/BELA ext": np.concatenate([all_lat[id_[3]], all_lat_ext[id_[3]]]),
# }

colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44',
          '#66CCEE','#AA3377', '#BBBBBB']
for (label, data), c in zip(datasets.items(), colors):
    # print(len(data))
    print(sum(data<60))
    print(sum(data<45))
    print(sum(data<30))
    plt.hist(
        data,
        nbins,
        color=c,
        density=False,
        label=label,
        histtype='step',
        # range=[0, 90],
        linewidth=1.5
    )


plt.xlabel(r"|$\phi$| [°]")
plt.xlim([0, 90])
plt.ylabel("Number of crossovers")
plt.yscale("log")
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color=colors[i], lw=1.5) for i,_ in enumerate(datasets.keys())]

# Get the current axes
ax = plt.gca()
ax.legend(custom_lines, datasets.keys())
# plt.legend(handlelength=3, handleheight=1)


# Grab the line handle from the histogram (step returns a Line2D object)
# handles, labels = ax.get_legend_handles_labels()

# Make legend use the line handle, not the default patch
# plt.legend(handles=handles, labels=labels)
#plt.legend(handles=handles, labels=labels, handlelength=2, handleheight=0)
fig_name = "hist_lat"
plt.savefig(f"{path}{fig_name}.png")
plt.savefig(f"{path}{fig_name}.pdf")